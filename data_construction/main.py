# Standard library imports
import argparse
import json
import logging
import os
import re
import traceback
from collections import Counter
from typing import List, Tuple
# Third-party imports
import jsonlines
from tqdm import tqdm
import tiktoken
import re
import difflib
# Local imports
from utils import config, cached, get_response, setup_logger, get_response_json, print_json, encode, decode, extract_json


def parse_args():
    parser = argparse.ArgumentParser(description='Construct dataset from source books')
    # parser.add_argument('--input', type=str, required=True,
    parser.add_argument('--input', type=str, default='dataset/original_books_from_gutenberg.jsonl',
                      help='Input jsonl file path containing books data')
    parser.add_argument('--output_dir', type=str, default='data',
                      help='Output directory path (default: data)')
    parser.add_argument('--num_workers', type=int, default=57,
                      help='Number of parallel workers (default: 57)')
    parser.add_argument('--model', type=str, default="gemini-2.5-pro",
                      help='Model to use for data construction (default: gemini-2.5-pro)')
    parser.add_argument('--candidate_model', type=str, default="claude-sonnet-4-5",
                      help='Another candidate model to use for data construction when the main model fails (default: claude-sonnet-4-5)')
    parser.add_argument('--regenerate', action='store_true',
                      help='Force regenerate data even if results already exist (default: False)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    return args

args = parse_args()

# Setup logger with custom configuration
# Console shows WARNING+, file shows DEBUG+
logger = setup_logger(__name__, 'data_construction/main.log', level=logging.DEBUG, console_level=logging.WARNING)

SCENE_FIELDS_TO_OMIT_FROM_PUBLIC_DATA = {
    'chapter_title',
    'first_sentence',
    'last_sentence',
    'state',
    'first_sentence_index',
    'last_sentence_index',
    'text',
}


def remove_private_scene_fields(data):
    """Remove extraction-only fields before writing public scene data."""
    if not isinstance(data, dict):
        return data

    data.pop('chapter_beginnings', None)
    for scene in data.get('scenes', []):
        if not isinstance(scene, dict):
            continue
        for field in SCENE_FIELDS_TO_OMIT_FROM_PUBLIC_DATA:
            scene.pop(field, None)
    return data


def value_to_str(v, indent=0):
    """Recursively convert a value to a readable string."""
    if isinstance(v, dict):
        parts = []
        for k2, v2 in v.items():
            parts.append(f"{'  ' * indent}**{k2}**\n{'  ' * indent}{value_to_str(v2, indent)}")
        return "\n\n".join(parts)
    elif isinstance(v, list):
        items = []
        for item in v:
            items.append(f"*   {value_to_str(item, indent)}")
        return "\n".join(items)
    elif v is None:
        return ""
    else:
        return str(v)


def dict_to_markdown(profile):
    """Convert a profile dict to a Markdown-formatted string. If already a str, return as-is."""
    if not isinstance(profile, dict):
        return profile
    # Handle the {'Current Profile': {...}} wrapper
    if len(profile) == 1 and "Current Profile" in profile:
        inner = profile["Current Profile"]
        if isinstance(inner, dict):
            profile = inner
    parts = []
    for key, value in profile.items():
        value_str = value_to_str(value)
        parts.append(f"**{key}**\n{value_str}")
    return "\n\n".join(parts)


def find_index(lst, key):
    """
    Find the index of a key in a list, returning -1 if not found.

    Args:
        lst: The list to search in
        key: The key to search for

    Returns:
        int: The index of the key if found, -1 if not found
    """
    try:
        return lst.index(key)
    except ValueError:
        return -1

@cached
def create_chunk_generator(book, chunk_size):
    """
    Generates chunks of text from a book while respecting token limits and chapter boundaries.

    Args:
        book (dict): A dictionary containing book information with 'content' and other fields
        chunk_size (int): Roughly the number of tokens per chunk

    Returns:
        list: A list of text chunks from the book, where each chunk is:
            - Limited to chunk_size if no chapters are detected
            - Between chunk_size/2 and 2*chunk_size if chapters are detected
            - Cleaned of copyright notices in the first chunk
            - Cleaned of excessive tabs if present

    The function handles books in two ways:
    1. For books without chapter markers: Splits into fixed-size chunks of chunk_size
    2. For books with chapters: Attempts to keep chapters together while staying within token limits
    """
    # Clean copyright notices from the beginning to avoid irrelevant text
    lines = book['content'].replace('\xa0', '\n').replace('\xad', '').split('\n')
    filtered_lines = []
    copyright_words = ['rights', 'reserved', 'reproduced', 'copyright', 'reproduce', 'permission']
    
    # Remove lines that are likely copyright notices (short lines with multiple copyright-related words)
    for line in lines:
        words = line.split()
        if len(words) < 50 and sum(word.lower() in copyright_words for word in words) > 1:
            continue
        filtered_lines.append(line)
    
    book['content'] = '\n'.join(filtered_lines)

    # Clean image markdown syntax (e.g., ![alt text](filename.jpg))
    book['content'] = re.sub(r'!\[.*?\]\([^)]*\)', '', book['content'])
    
    # Check and clean excessive tabs that may interfere with text processing
    def has_excessive_tabs(content, threshold=0.05):
        tab_count = content.count('\t')
        return (tab_count / len(content)) > threshold
    
    if has_excessive_tabs(book['content']):
        book['content'] = book['content'].replace('\t', '')

    # Try to split book into chapters using split_book utility
    from split import split_book
    chapters = split_book(book)

    results = []
    
    # Unified chunking logic: split text into chunks of size between chunk_size/2 and 2*chunk_size
    # Try to respect chapter boundaries when available, but always respect sentence boundaries
    
    # Prepare the text segments to process
    if isinstance(chapters, list):
        # If chapters detected, use them as base segments
        segments = [chapter['content'] for chapter in chapters]
        # Track chapter indices for each segment
        segment_chapter_indices = list(range(len(chapters)))
    else:
        # If no chapters, treat entire book as one segment
        segments = [chapters]
        segment_chapter_indices = [None]  # No chapter info
    
    # Process segments into chunks
    current_text = ""
    current_tokens = 0
    current_chapter_indices = []  # Track which chapters are in current chunk
    
    for i_segment, segment in enumerate(segments):
        segment_tokens = len(encode(segment))
        chapter_idx = segment_chapter_indices[i_segment]
        
        # If adding this segment keeps us within 2*chunk_size, accumulate it
        if current_tokens > 0 and current_tokens + segment_tokens <= chunk_size * 2:
            current_text += segment
            current_tokens += segment_tokens
            if chapter_idx is not None and chapter_idx not in current_chapter_indices:
                current_chapter_indices.append(chapter_idx)
            continue
        
        # If current accumulated text is large enough (>= chunk_size/2), save it as a chunk
        if current_tokens >= chunk_size // 2:
            chapter_info = f" (chapters: {current_chapter_indices})" if current_chapter_indices else ""
            logger.debug(f'Chunk {len(results) + 1} tokens: {current_tokens}{chapter_info}')
            results.append(current_text)
            current_text = segment
            current_tokens = segment_tokens
            current_chapter_indices = [chapter_idx] if chapter_idx is not None else []
        # If current text is too small but adding segment exceeds limit, save current and start new
        elif current_tokens > 0:
            current_text += segment
            current_tokens += segment_tokens
            if chapter_idx is not None and chapter_idx not in current_chapter_indices:
                current_chapter_indices.append(chapter_idx)
        # If current is empty, start with this segment
        else:
            current_text = segment
            current_tokens = segment_tokens
            current_chapter_indices = [chapter_idx] if chapter_idx is not None else []
        
        # If accumulated text exceeds 2*chunk_size, split it at sentence boundaries
        while current_tokens > chunk_size * 2:
            tokens = encode(current_text)
            
            # Try to find a good split point around chunk_size
            target_split = min(chunk_size, len(tokens))
            chunk_candidate = decode(tokens[:target_split])
            
            # Find the last sentence ending
            sentence_endings = ['. ', '! ', '? ', '." ', '!" ', '?" ', '.\n', '!\n', '?\n']
            last_sentence_end = -1
            
            for ending in sentence_endings:
                pos = chunk_candidate.rfind(ending)
                if pos > last_sentence_end:
                    last_sentence_end = pos + len(ending)
            
            # Use sentence boundary if found and it's at least chunk_size/2
            if last_sentence_end > 0:
                text_before_boundary = chunk_candidate[:last_sentence_end]
                tokens_before_boundary = encode(text_before_boundary)
                
                if len(tokens_before_boundary) >= chunk_size // 2:
                    chapter_info = f" (chapters: {current_chapter_indices})" if current_chapter_indices else ""
                    logger.debug(f'Chunk {len(results) + 1} tokens: {len(tokens_before_boundary)}{chapter_info}')
                    results.append(text_before_boundary)
                    current_text = current_text[last_sentence_end:]
                    current_tokens = len(encode(current_text))
                    # Keep chapter indices as they're still in the remaining text
                    continue
            
            # If no good sentence boundary, split at chunk_size
            chapter_info = f" (chapters: {current_chapter_indices})" if current_chapter_indices else ""
            logger.debug(f'Chunk {len(results) + 1} tokens: {target_split}{chapter_info}')
            results.append(chunk_candidate)
            current_text = decode(tokens[target_split:])
            current_tokens = len(encode(current_text))
            # Keep chapter indices as they're still in the remaining text
    
    # Add any remaining content as final chunk
    if current_tokens > 0:
        chapter_info = f" (chapters: {current_chapter_indices})" if current_chapter_indices else ""
        logger.debug(f'Chunk {len(results) + 1} tokens: {current_tokens}{chapter_info}')
        results.append(current_text)

    book['content'] = '\n'.join(results)
    return results, book


def ngram_jaccard_similarity(text1, text2, n=3):
    """Calculate the Jaccard similarity between two texts using n-grams.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare 
        n (int, optional): Size of n-grams. Defaults to 3.
    
    Returns:
        float: Jaccard similarity score between 0 and 1, where 1 means identical texts
              and 0 means completely different texts.
    """
    def ngrams(tokens, n):
        """Generate n-grams from a sequence of tokens.
        
        Args:
            tokens (list): List of tokens
            n (int): Size of n-grams
        Returns:
            list: List of n-gram tuples
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def jaccard_similarity(set1, set2):
        """Calculate Jaccard similarity between two sets.
        
        Args:
            set1 (set): First set
            set2 (set): Second set
        Returns:
            float: Jaccard similarity score
        """
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    # Tokenize the input texts into sequences of tokens
    tokens1 = encode(text1)
    tokens2 = encode(text2)
    
    # Generate sets of n-grams from the token sequences
    ngrams1 = set(ngrams(tokens1, n))
    ngrams2 = set(ngrams(tokens2, n))
    
    # Calculate and return the Jaccard similarity between the n-gram sets
    return jaccard_similarity(ngrams1, ngrams2)

@cached
def find_best_match_passage(candidates, target, n=3, threshold=0.3):
    """Find the best matching passage from a list of candidates compared to a target text. These texts are generally LLM-synthesized summaries. Hence, we focus on their semantic similarity.

    Uses n-gram Jaccard similarity to compare texts and find the closest match.
    
    Args:
        candidates (list): List of candidate passages to search through
        target (str or dict): Target text to match against
        n (int, optional): Size of n-grams to use for comparison. Defaults to 3.
        threshold (float, optional): Minimum similarity score required to consider a match.
                                   Defaults to 0.3.
    
    Returns:
        int: Index of best matching passage if score >= threshold, -1 if no good match found
    """
    best_match = None  # Index of current best matching passage
    best_score = 0     # Highest similarity score found so far

    # Handle case where inputs are dictionaries by converting to strings
    if isinstance(candidates, list) and isinstance(target, dict) and isinstance(candidates[0], dict):
        target = str(target)
        candidates = [str(c) for c in candidates]

    # Compare target against each candidate passage
    for i, candidate in enumerate(candidates):
        score = ngram_jaccard_similarity(target, candidate, n)
        if score >= best_score:
            best_score = score
            best_match = i
    
    # Return best match if it meets threshold, otherwise return -1
    if best_score >= threshold:
        logger.debug(f"Best match: \nInput: {target}\nOutput: {candidates[best_match]}\nScore: {best_score}")
        return best_match
    else:
        return -1


@cached
def find_best_match_sentence(chunk, target, threshold=0.5):
    """Find the best matching sentence from a chunk of text or list of sentences compared to a target sentence. These sentences are generally exact sentences from the book, so we focus on their string similarity.
    
    Uses SequenceMatcher to calculate string similarity ratios between sentences to find the closest match.
    
    Args:
        chunk (str or list): Input text chunk or list of sentences to search through
        target (str): Target sentence to match against
        threshold (float, optional): Minimum similarity score required to consider a match.
                                   Defaults to 0.5.
    
    Returns:
        str or None: Best matching sentence if score >= threshold, None if no good match found
                    or if target is None/invalid
    """
    # Return None for invalid target inputs
    if target == 'None' or target is None:
        return None

    # Split chunk into sentences if it's a string, otherwise use as-is if it's a list
    if isinstance(chunk, str):
        # Use a more sophisticated sentence splitting approach
        # First, do a basic split on sentence endings
        raw_sentences = re.split(r'(?<=[.!?。！？])\s+', chunk)
        
        # Common abbreviations that shouldn't trigger sentence breaks
        abbreviations = ['Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'vs', 'etc', 'Inc', 'Ltd', 'Co']
        
        sentences = []
        i = 0
        while i < len(raw_sentences):
            current = raw_sentences[i]
            
            # Check if this might be a false split (abbreviation or initial)
            # Conditions for merging with next sentence:
            # 1. Current segment is very short (< 10 chars) - likely an abbreviation or initial
            # 2. Current segment ends with a known abbreviation
            # 3. Current segment ends with a single letter followed by period (initial like "J.")
            should_merge = False
            
            if i < len(raw_sentences) - 1:  # Not the last segment
                # Check if too short (likely abbreviation or initial)
                if len(current.strip()) < 10:
                    should_merge = True
                # Check if ends with known abbreviation
                elif any(current.strip().endswith(abbr + '.') for abbr in abbreviations):
                    should_merge = True
                # Check if ends with single letter + period (initial like "J." or "K.")
                elif re.search(r'\b[A-Z]\.$', current.strip()):
                    should_merge = True
            
            if should_merge:
                # Merge with next segment
                current = current + ' ' + raw_sentences[i + 1]
                i += 2
            else:
                i += 1
            
            if current.strip():  # Only add non-empty sentences
                sentences.append(current)
    else: # chunk is a list
        assert isinstance(chunk, list)  
        sentences = chunk

    # Initialize variables to track best match
    best_match = 0
    best_score = 0
    
    # Compare target against each sentence
    for i, sentence in enumerate(sentences):
        # Calculate similarity ratio between target and current sentence
        score = difflib.SequenceMatcher(None, target, sentence).ratio()
        
        # Update best match if current score is higher
        if score > best_score:
            best_score = score
            best_match = sentence
    
    # Log the matching results
    logger.debug(f"Best match: \nInput: {target}\nOutput: {best_match}\nScore: {best_score}")

    # Return best match if it meets threshold, otherwise return None
    if best_score >= threshold:
        return best_match
    else:
        return None

def split_into_sentences(text):
    """Split text into sentences, handling common abbreviations and initials.
    
    Args:
        text (str): Input text to split
    
    Returns:
        list: List of sentences
    """
    raw_sentences = re.split(r'(?<=[.!?\u3002\uff01\uff1f])\s+', text)
    
    abbreviations = ['Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'vs', 'etc', 'Inc', 'Ltd', 'Co']
    
    sentences = []
    i = 0
    while i < len(raw_sentences):
        current = raw_sentences[i]
        should_merge = False
        
        if i < len(raw_sentences) - 1:
            if len(current.strip()) < 10:
                should_merge = True
            elif any(current.strip().endswith(abbr + '.') for abbr in abbreviations):
                should_merge = True
            elif re.search(r'\b[A-Z]\.$', current.strip()):
                should_merge = True
        
        if should_merge:
            current = current + ' ' + raw_sentences[i + 1]
            i += 2
        else:
            i += 1
        
        if current.strip():
            sentences.append(current)
    
    return sentences

def extract_from_chunk(book, i_c, chunk, truncated_plots=None):
    """
    Extract and process scene information from a chunk of book text.
    
    This function analyzes a chunk of text to identify chapter beginnings, scenes, interactions,
    and other narrative elements. It uses an LLM to generate structured information about the text.

    Args:
        book (dict): Dictionary containing book metadata including title and author
        i_c (int): Chunk index
        chunk (str): Text content of the current chunk to analyze
        truncated_plots (list, optional): List of incomplete scenes from previous chunk that need to be finished

    Returns:
        tuple: Contains:
            - chapter_beginnings (list): List of identified chapter starts
            - scenes (list): Extracted scene information including summaries, characters, interactions
            - remaining_chunk (str): Unused portion of chunk to process in next iteration
            
    The function generates a detailed prompt for the LLM that requests:
    1. Chapter beginning identification
    2. Scene extraction and analysis
    3. Interaction reconstruction
    4. Character motivation analysis
    5. Next chunk starting point determination
    """
    logger.info(f"Extracting plots from chunk for book: {book['title']}")

    # Create deep copy of truncated scenes and remove unnecessary fields
    import copy
    if truncated_plots:
        truncated_plots = copy.deepcopy(truncated_plots)
        for scene in truncated_plots:
            # Remove fields that will be recalculated later
            scene.pop('first_sentence_index', None)
            scene.pop('last_sentence_index', None)
            scene.pop('text', None)
    
    # Construct the prompt for the LLM
    prompt = f"""
Extract structured narrative information from this book chunk. Extract ALL content completely.

## TASKS

### 1. Identify Chapter Beginnings
Record exact first line if a new chapter starts in this chunk.

### 2. Extract ALL Scenes (Chronological)
- Extract **every** scene, event, or narrative segment (major and minor)
- Provide: first sentence, last sentence, chapter title, prominence (1-100), state ("finished"/"truncated")
- If truncated scenes are provided from previous chunk, extend them with current content

### 3. Extract Complete Scenes
**Scene Structure:**
- **scenario**: Time, location, atmosphere, background (detailed, exclude details in interactions)
- **interactions**: ALL character turns (15-20+ per scene)
- **summary**: Comprehensive scene summary
- **key_characters**: Names, descriptions, experiences, and motivations
  - Derived AFTER extracting all interactions: collect every named individual who appears in interactions, then fill in their info

**Interaction Format: [thought] speech (action)**
- **[thought]**: Internal perspective, emotions, motivations (REQUIRED, can repeat; but every interaction MUST start with thought, instead of speech or action)
  - Based on original text OR reasonably inferred from actions, avoid over-interpretation
- **speech**: Exact spoken words (optional, can repeat)
- **(action)**: Body language, facial expressions, tone, pauses, gestures, physical actions (optional, can repeat)
  - NOT simple tags like "(said/replied X)"

**Examples:**
- "[I wonder what she means]"
- "[This makes me uncomfortable] (fidgets with hands)"
- "[I need to be careful] Perhaps we should reconsider"
- "[She seems upset] Are you alright? (reaches out gently)"
- "[I can't believe this is happening] This is outrageous! (slams fist) [I need to calm down] But let's discuss rationally."
- "[I need to investigate] (walks across room and examines painting)"

**Extraction Rules:**
1. Extract from BOTH dialogue scenes AND narrative descriptions
   - Dialogue: Extract interactions with thoughts/speeches/actions
   - Narrative: Convert summarized actions to interaction format
    - Example: "The Smith family had a wonderful evening." → [Everything is perfect tonight] (have a wonderful evening)
    - Example: "The children walked into room together" → [We need to stay together] (walk into room together)
2. Extract ALL interactions (15-20+ per scene minimum)
3. Segment or supplement original text so each interaction's content is from the corresponding character(s)' perspective
   - Ensure content matches the subject in "characters" field
4. Merge consecutive turns from same character into **ONE** interaction
   - Don't worry about long interactions; use multiple [thought]/speech/(action) to represent them
5. Use "Environment" as character for atmosphere/weather/sound/... (exclude character's active thoughts/observations/actions)
6. Each character in "characters" field MUST be a specific individual's name 
    — **NEVER use vague group labels** like "other people", "all guests", "the crowd", "everyone", or pronouns
    - When multiple named individuals act together, list ALL their names explicitly: "The Smith family" → ["Mr. Smith", "Mrs. Smith", ...]
    - If the character group consists of minor/insignificant characters (e.g., unnamed passengers on a bus) and animals (e.g., horses on a farm) not central to the plot, do NOT list them as characters, instead incorporate their actions/presence into the `Environment` description
7. Use exact text from book; convert third-person narrative to interaction format
8. Match the chunk's language

**Extraction Order (IMPORTANT):**
1. First identify all **key_characters** in the scene — every specific named individual who appears in the scene (must be a specific individual's name, no vague group labels)
   - If a Character Group is composed of named individuals, expand it into each individual's name separately
2. Then extract all **interactions** — each interaction's `characters` field MUST only contain names from the scene's key_characters list above

### 4. Identify Next Chunk Start
Output None if: last scene is truncated OR last scene finishes exactly at chunk end
Otherwise: output first sentence of next unprocessed storyline

## OUTPUT FORMAT (JSON)

{{
    "chapter_beginnings": [
        {{"beginning_sentence": "exact first line"}}
    ],
    "scenes": [
        // Extend the truncated scenes from previous chunk, if any
        {{
            ...
        }},
        {{
            "chapter_title": "chapter name or None",
            "first_sentence": "exact first sentence of this scene",
            "last_sentence": "exact last sentence of this scene",
            "prominence": "1-100",
            "scenario": "detailed scene setup",
            "interactions": [
                {{
                    "characters": ["name 1"] or ["name 1", "name 2", ...] or ["Environment"] (Note: 'characters' is always a list. Most interactions have one character, but group actions include multiple characters.),
                    "content": "[thought] speech (action) ... (MUST be from the perspective of the character(s) listed in 'characters')"
                }}
            ],
            "summary": "scene summary",
            "key_characters": [
                {{
                    "name": "full name without title",
                    "description": "character description before this scene (~20 words)",
                    "experience": "role, thoughts, behaviors, development in this scene (~30 words)",
                    "motivation": "thoughts/feelings/goals before the above interactions"
                }}
            ],
            "state": "finished" or "truncated"
        }}
    ],
    "next_chunk_start": "first sentence of the next storyline or None"
}}

## REQUIREMENTS
1. Output MUST strictly follow the JSON format above 
    — the top-level object MUST contain exactly these keys: `chapter_beginnings`, `scenes`, `next_chunk_start`. Do NOT wrap in extra keys or change the structure.
2. Valid JSON with escaped quotes
3. Full character names without titles
4. Chronological order
5. Extract ALL content - no skipping
6. Use exact book text when available
7. Scene key_characters = **ALL** named individuals who appear in the scene (no duplicates)
   - MUST be specific individual names — no vague group labels
   - If a Character Group is composed of named individuals, list each individual separately
   - Each key_character MUST have: name, description, experience, motivation
   - Interaction `characters` MUST only reference names already in key_characters; unnamed characters go into Environment

## INPUT

Book: {book['title']}
Author: {book['author']}

Chunk:
{chunk}

Truncated Scene from Previous Chunk:
{json.dumps(truncated_plots, ensure_ascii=False, indent=2) if truncated_plots else "None"}
"""
    
    # Example format for character utterances in conversations
    # "[My father's words fill me with awe, but I still feel uneasy.] 
    # (Nods seriously, but with a slight frown remaining) 
    # I understand, Father. Responsibility is important. But… is killing really necessary? 
    # (A flash of compassion in his eyes)
    # If someone has done something wrong, can't we give them a chance to make amends?"

    def parse_response(response, chunk, book, **kwargs):
        """
        Parse and validate the LLM response, extracting structured scene information.
        
        Args:
            response: Raw LLM response to parse
            chunk: Original text chunk for reference
            book: Book metadata
            **kwargs: Additional keyword arguments
            
        Returns:
            tuple or bool: (chapter_beginnings, scenes, remaining_chunk) if successful, False if failed
        """
        if not response:
            return False
        
        try:
            # Handle different response formats
            if isinstance(response, dict) and 'first_sentence' in response:
                # LLM returned a single scene object, wrap into full structure
                response = {
                    'chapter_beginnings': [],
                    'scenes': [response],
                    'next_chunk_start': None
                }
            elif isinstance(response, list) and len(response) > 0 and 'first_sentence' in response[0]:
                # LLM returned a scenes array, wrap into full structure
                response = {
                    'chapter_beginnings': [],
                    'scenes': response,
                    'next_chunk_start': None
                }

            try:
                chapter_beginnings = response['chapter_beginnings']
            except:
                print(f"Error: {response}")

            scenes = []

            # Process next chunk starting point
            next_chunk_start = response.get('next_chunk_start')

            if next_chunk_start:
                next_chunk_start = find_best_match_sentence(chunk, next_chunk_start)

                if next_chunk_start:
                    remaining_chunk = chunk[chunk.index(next_chunk_start):]
                else:
                    remaining_chunk = ''
            else:
                remaining_chunk = ''
            
            # Process each scene from the response
            for unprocessed_scene in response['scenes']:
                
                chapter_title = unprocessed_scene.get('chapter_title')

                # Extract first sentence of first_sentence and last sentence of last_sentence
                if unprocessed_scene.get('first_sentence'):
                    fs_sentences = split_into_sentences(unprocessed_scene['first_sentence'])
                    unprocessed_scene['first_sentence'] = fs_sentences[0] if fs_sentences else unprocessed_scene['first_sentence']
                if unprocessed_scene.get('last_sentence'):
                    ls_sentences = split_into_sentences(unprocessed_scene['last_sentence'])
                    unprocessed_scene['last_sentence'] = ls_sentences[-1] if ls_sentences else unprocessed_scene['last_sentence']

                # Find exact matches for scene boundaries in original text
                unprocessed_scene['first_sentence'] = find_best_match_sentence(chunk, unprocessed_scene['first_sentence'])
                unprocessed_scene['last_sentence'] = find_best_match_sentence(chunk, unprocessed_scene['last_sentence'])

                first_sentence, last_sentence = unprocessed_scene['first_sentence'], unprocessed_scene['last_sentence']
                
                # If key_characters is None, set it to an empty list
                if unprocessed_scene.get('key_characters') is None:
                    unprocessed_scene['key_characters'] = []
                
                # Validate key_characters structure
                if not all(['name' in c for c in unprocessed_scene['key_characters']]):
                    logger.warning(f"Scene key_characters missing 'name' field, rejecting response")
                    return False
                
                # Check for duplicate characters in scene key_characters
                scene_char_names = [c['name'] for c in unprocessed_scene['key_characters']]
                if len(scene_char_names) != len(set(scene_char_names)):
                    logger.warning(f"Scene has duplicate characters in key_characters, rejecting response")
                    return False
                
                # Collect all characters who initiate interactions (exclude "Environment")
                interaction_characters = set()
                for interaction in unprocessed_scene.get('interactions', []):
                    if 'characters' in interaction:
                        chars = interaction['characters'] if isinstance(interaction['characters'], list) else [interaction['characters']]
                        for char in chars:
                            if char != "Environment":
                                interaction_characters.add(char)
                
                # Verify all interaction characters are in scene's key_characters
                scene_key_char_names = set(scene_char_names)
                missing_chars = interaction_characters - scene_key_char_names
                if missing_chars:
                    logger.warning(f"Scene key_characters missing interaction characters: {missing_chars}, rejecting response")
                    return False
                    
                # Validate summary is a non-empty string (required for find_best_match_passage)
                summary = unprocessed_scene.get('summary')
                if not summary or not isinstance(summary, str):
                    logger.warning(f"Scene summary is missing or not a string: {repr(summary)}, rejecting response")
                    return False

                # Create structured scene object
                scene = {
                    'chapter_title': chapter_title,
                    'first_sentence': first_sentence,
                    'last_sentence': last_sentence,
                    'prominence': unprocessed_scene.get('prominence'),
                    'scenario': unprocessed_scene.get('scenario', ''),
                    'interactions': unprocessed_scene.get('interactions', []),
                    'summary': summary,
                    'key_characters': unprocessed_scene['key_characters'],
                    'state': unprocessed_scene['state']
                }

                scenes.append(scene)

            return chapter_beginnings, scenes, remaining_chunk
        
        except Exception as e:
            logger.error(f"Error processing chunk for book {book['title']}: {e}, {traceback.format_exc()}")
            logger.error(f"Prompt: {prompt}")
            logger.error(f"Response: {json.dumps(response, ensure_ascii=False, indent=2) if isinstance(response, (dict, list)) else response}")
            return False
    # Get and parse LLM response
    response = get_response_json([extract_json, parse_response], model=args.model, messages=[{"role": "user", "content": prompt}], book=book, chunk=chunk, fix_truncated_json=True)

    return response

def extract(book, chunk_size=1500):
    """Process a book by splitting it into chunks and extracting structured information.

    This function processes a book by:
    1. Splitting the book text into chunks of specified size
    2. Extracting chapter beginnings, scenes and interactions from each chunk
    3. Handling truncated scenes that span multiple chunks by merging them
    4. Saving the extracted results to a JSON file

    Args:
        book (dict): Book data containing 'title', 'author', and 'content'
        chunk_size (int, optional): Roughly the number of tokens per chunk. Defaults to 1500.

    Returns:
        dict: Extracted results containing:
            - chapter_beginnings: List of chapter names (start locations)
            - scenes: List of extracted scenes with interactions
            - fail: List of failed chunks with their truncated_plots and fail_to_parse_response
    """
    # Set up save path and skip if already processed
    save_dir = f'{args.output_dir}/extracted'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{book["title"]}.json'
    if os.path.exists(save_path) and not args.regenerate:
        return 

    # Set up cache path
    from utils import set_cache_path
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')

    # Create generator to iterate through book chunks
    chunk_generator, book = create_chunk_generator(book, chunk_size)
    
    # Convert generator to list to get total count for progress bar
    chunks = list(chunk_generator)
    total_chunks = len(chunks)

    # Initialize results structure
    results = {
        'chapter_beginnings': [],
        'scenes': [],
        'fail': [],
    }

    # Track state between chunks
    remaining_chunk = ''  # Text carried over from previous chunk
    truncated_plots = []  # Scenes that continue into next chunk
    
    # Process each chunk with progress bar
    for i, chunk in enumerate(tqdm(chunks, desc=f"Extracting {book['title']}", leave=False)):
        
        logger.info(f"Processing chunk {i} with {len(encode(chunk))} tokens")

        # Extract information from current chunk
        current_input_chunk = remaining_chunk + '\n' + chunk
        response = extract_from_chunk(book, i, current_input_chunk, truncated_plots)

        # Handle the response
        if response:
            if isinstance(response, tuple) and len(response) == 3:
                # Successful extraction
                chapter_beginnings, scenes, remaining_chunk = response 
            else:
                # Failed extraction
                chapter_beginnings, scenes, remaining_chunk = [], [], ''
                results['fail'].append({
                    'chunk': current_input_chunk,
                    'truncated_plots': truncated_plots,
                    'fail_to_parse_response': response.get('fail_to_parse_response')
                })
        else:
            # No response
            chapter_beginnings, scenes, remaining_chunk = [], [], ''
            results['fail'].append({
                'chunk': current_input_chunk,
                'truncated_plots': truncated_plots,
                'fail_to_parse_response': None
            })

        # Merge truncated scenes from previous chunk with current scenes
        for u_scene in truncated_plots:
            # Find matching scene in current chunk (by summary similarity)
            # Guard against None summaries to avoid TypeError in encode()
            u_summary = u_scene.get('summary') or ''
            candidate_summaries = [s.get('summary') or '' for s in scenes]
            idx = find_best_match_passage(candidate_summaries, u_summary) if u_summary and any(candidate_summaries) else -1

            if idx != -1:
                # Found matching scene - merge them
                # Keep the first_sentence from the previous chunk (u_scene)
                # and the last_sentence from the current chunk (scenes[idx])
                scenes[idx]['first_sentence'] = u_scene['first_sentence']
                scenes[idx]['chapter_title'] = u_scene.get('chapter_title') or scenes[idx].get('chapter_title')
                scenes[idx]['prominence'] = u_scene.get('prominence') or scenes[idx].get('prominence')
  
                # Merge interactions: keep old interactions and append new ones
                old_interactions = u_scene.get('interactions', [])
                new_interactions = scenes[idx].get('interactions', [])
                # Avoid duplicating interactions already in new_interactions
                merged_interactions = old_interactions + [
                    inter for inter in new_interactions if inter not in old_interactions
                ]
                scenes[idx]['interactions'] = merged_interactions

                # Merge key_characters: prefer new scene's key_characters (more complete)
                # but keep any characters from old scene not in new scene
                new_char_names = {c['name'] for c in scenes[idx].get('key_characters', [])}
                for old_char in u_scene.get('key_characters', []):
                    if old_char['name'] not in new_char_names:
                        scenes[idx]['key_characters'].append(old_char)
            else:
                # No matching scene found - mark as finished
                u_scene['state'] = 'finished'
                results['scenes'].append(u_scene)

        # Separate/Update finished and truncated scenes
        finished_scenes = [scene for scene in scenes if scene['state'] == 'finished']
        truncated_plots = [scene for scene in scenes if scene['state'] == 'truncated']
        
        # Add to results
        results['chapter_beginnings'].extend(chapter_beginnings)
        results['scenes'].extend(finished_scenes)

    # Finish any remaining truncated scenes
    for u_scene in truncated_plots:
        u_scene['state'] = 'finished'
        results['scenes'].append(u_scene)
    
    # Index each scene's first and last sentence in the whole book content and extract text
    book_content = book['content']
    for scene in results['scenes']:
        # Match first_sentence and last_sentence independently in book_content
        matched_first = find_best_match_sentence(book_content, scene.get('first_sentence')) if scene.get('first_sentence') else None
        matched_last = find_best_match_sentence(book_content, scene.get('last_sentence')) if scene.get('last_sentence') else None

        first_sentence_index = book_content.find(matched_first) if matched_first else -1
        last_sentence_index = book_content.find(matched_last) + len(matched_last) if matched_last and book_content.find(matched_last) != -1 else -1

        scene['first_sentence_index'] = first_sentence_index
        scene['last_sentence_index'] = last_sentence_index

        # Extract text between first and last sentence if both are found
        if first_sentence_index != -1 and last_sentence_index != -1 and first_sentence_index <= last_sentence_index:
            scene['text'] = book_content[first_sentence_index:last_sentence_index]
        else:
            scene['text'] = ''
    
    # Sort all scenes by their first sentence index to ensure narrative order
    # results['scenes'] = sorted(results['scenes'], key=lambda x: x.get('first_sentence_index', x.get('last_sentence_index', float('inf'))))
    
    # Save results
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

count_nth_generation = {i: 0 for i in range(7)}

def restore_from_cache(book):
    """
    As we typically encounter issues during the extraction process, this function restores some previously extracted plot data from cache.
    
    This function loads cached responses from extraction LLMs, processes them into the regular data format, and merges them with the results of extract(). 

    Args:
        book (dict): Book data containing title, author, and content

    Returns:
        str: Book name
    """
    # Load existing extracted results

    save_dir = f'{args.output_dir}/extracted'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{book["title"]}.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    save_path = f'{save_dir}/{book["title"]}.json'

    # Skip if already processed
    if os.path.exists(save_path) and not args.regenerate:
       return book['title']

    # Load cached API responses
    import pickle
    cache_path = f'.cache/cache_{book["title"]}.pkl'
    if not os.path.exists(cache_path):
        logger.warning(f"Cache file not found for {book['title']}, skipping restore_from_cache.")
        return book['title']

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    # Get only the get_response cache entries
    keys = [ k for k in cache.keys() if k[0] == 'get_response' ]

    global count_nth_generation

    fail_prompts = []
    responses = {}

    # Generate chunks from book content
    chunk_generator, book = create_chunk_generator(book, chunk_size=1500)
    chunks = [chunk for chunk in chunk_generator]

    # Process each cached response with progress bar
    for key, value in tqdm(cache.items(), desc=f"Restoring {book['title']}", leave=False):
        if key[0] == 'get_response':
            # Extract kwargs from cache key
            dict_string = key[-1][11:-1]
            import ast
            parsed_list = ast.literal_eval(dict_string)
            restored_kwargs = dict(parsed_list)

            # Only process responses for plot extraction prompts
            if restored_kwargs['model'] == 'claude-3-5-sonnet-20240620' and restored_kwargs['messages'][0]['content'].startswith("\nBased on the provided book chunk, complete the following tasks:\n\n1. Recognize chapter beginnings if"):
                # Verify book title matches
                if not restored_kwargs['book']['title'] == book['title']:
                    logger.info(f"Warning: {restored_kwargs['book']['title']} != {book['title']}")
                    continue

                # Track generation attempts
                nth_generation = restored_kwargs['nth_generation']
                count_nth_generation[nth_generation] += 1

                # Store response
                prompt = restored_kwargs['messages'][0]['content']
                responses.setdefault(prompt, {})
                responses[prompt][nth_generation] = value

                # Track failed prompts (those that needed max retries)
                if nth_generation == 5:
                    fail_prompts.append(prompt)

    fetched_plots = []

    # Process failed prompts to extract any valid plots
    for prompt in fail_prompts:
        for nth_generation in range(6):
            if nth_generation in responses[prompt]:
                response = responses[prompt][nth_generation]
                
                # Check if response contains all required fields
                required_fields = ["chapter_beginnings", "scenes", "first_sentence", "last_sentence", "summary", "key_characters"]
                if all(field in str(response) for field in required_fields):
                    # Extract JSON from potentially truncated response
                    from utils import extract_json
                    response = extract_json(response, post_fix_truncated_json=True)

                    if response is None:
                        continue

                    # Helper function to parse response and extract scenes
                    def parse_response(response, chunk, book, **kwargs):
                        if not response:
                            return False
                        
                        try:
                            # Normalize response format
                            if isinstance(response, dict) and 'first_sentence' in response:
                                # LLM returned a single scene object, wrap into full structure
                                response = {
                                    'chapter_beginnings': [],
                                    'scenes': [response],
                                    'next_chunk_start': None
                                }
                            elif isinstance(response, list) and len(response) > 0 and 'first_sentence' in response[0]:
                                # LLM returned a scenes array, wrap into full structure
                                response = {
                                    'chapter_beginnings': [],
                                    'scenes': response,
                                    'next_chunk_start': None
                                }

                            try:
                                chapter_beginnings = response['chapter_beginnings']
                            except:
                                print(f"Error: {response}")

                            scenes = []

                            # Handle remaining chunk logic
                            next_chunk_start = response.get('next_chunk_start')

                            if next_chunk_start:
                                next_chunk_start = find_best_match_sentence(chunk, next_chunk_start)

                                if next_chunk_start:
                                    remaining_chunk = chunk[chunk.index(next_chunk_start):]
                                else:
                                    remaining_chunk = ''
                            else:
                                remaining_chunk = ''
                            
                            # Process each scene in the response
                            for unprocessed_scene in response['scenes']:
                                chapter_title = unprocessed_scene.get('chapter_title')

                                # Extract first sentence of first_sentence and last sentence of last_sentence
                                if unprocessed_scene.get('first_sentence'):
                                    fs_sentences = split_into_sentences(unprocessed_scene['first_sentence'])
                                    unprocessed_scene['first_sentence'] = fs_sentences[0] if fs_sentences else unprocessed_scene['first_sentence']
                                if unprocessed_scene.get('last_sentence'):
                                    ls_sentences = split_into_sentences(unprocessed_scene['last_sentence'])
                                    unprocessed_scene['last_sentence'] = ls_sentences[-1] if ls_sentences else unprocessed_scene['last_sentence']

                                # Match first and last sentences
                                unprocessed_scene['first_sentence'] = find_best_match_sentence(chunk, unprocessed_scene['first_sentence'], threshold=0.6)
                                unprocessed_scene['last_sentence'] = find_best_match_sentence(chunk, unprocessed_scene['last_sentence'], threshold=0.6)

                                first_sentence, last_sentence = unprocessed_scene['first_sentence'], unprocessed_scene['last_sentence']

                                # Ensure key_characters is not None
                                if unprocessed_scene.get('key_characters') is None:
                                    unprocessed_scene['key_characters'] = []
                                
                                # Validate key_characters structure
                                if not all(['name' in c for c in unprocessed_scene['key_characters']]):
                                    logger.warning(f"Scene key_characters missing 'name' field, rejecting response")
                                    return False
                                
                                # Check for duplicate characters in scene key_characters
                                scene_char_names = [c['name'] for c in unprocessed_scene['key_characters']]
                                if len(scene_char_names) != len(set(scene_char_names)):
                                    logger.warning(f"Scene has duplicate characters in key_characters, rejecting response")
                                    return False
                                
                                # Collect all characters who initiate interactions (exclude "Environment")
                                interaction_characters = set()
                                for interaction in unprocessed_scene.get('interactions', []):
                                    if 'characters' in interaction:
                                        chars = interaction['characters'] if isinstance(interaction['characters'], list) else [interaction['characters']]
                                        for char in chars:
                                            if char != "Environment":
                                                interaction_characters.add(char)
                                
                                # Verify all interaction characters are in scene's key_characters
                                scene_key_char_names = set(scene_char_names)
                                missing_chars = interaction_characters - scene_key_char_names
                                if missing_chars:
                                    logger.warning(f"Scene key_characters missing interaction characters: {missing_chars}, rejecting response")
                                    return False

                                # Create scene object
                                scene = {
                                    'chapter_title': chapter_title,
                                    'first_sentence': first_sentence,
                                    'last_sentence': last_sentence,
                                    'prominence': unprocessed_scene.get('prominence'),
                                    'scenario': unprocessed_scene.get('scenario', ''),
                                    'interactions': unprocessed_scene.get('interactions', []),
                                    'summary': unprocessed_scene.get('summary', ''),
                                    'key_characters': unprocessed_scene['key_characters'],
                                    'state': unprocessed_scene['state']
                                }

                                scenes.append(scene)

                            return chapter_beginnings, scenes, remaining_chunk
                        
                        except Exception as e:
                            logger.error(f"Error processing chunk for book {book['title']}: {e}, {traceback.format_exc()}")
                            logger.error(f"Prompt: {prompt}")
                            logger.error(f"Response: {json.dumps(response, ensure_ascii=False, indent=2) if isinstance(response, (dict, list)) else response}")
                            return False
                    
                    # Extract chunk from prompt
                    chunk = prompt.split('Truncated Scene from Previous Chunk:')[0].split('Chunk:')[-1].strip(' \n')

                    # Parse response to get scenes
                    res = parse_response(response, chunk, book)

                    if res :
                        chapter_beginnings, scenes, remaining_chunk = res
                    else:
                        continue

                    # Process extracted scenes
                    for scene in scenes:
                        scene['state'] = 'finished'

                    fetched_plots.extend(scenes)
                    break 
    
    # Merge all scenes
    logger.info(f'Number of Original Scenes: {len(results["scenes"])}, Fetched New Scenes: {len(fetched_plots)}, Total Scenes: {len(results["scenes"]) + len(fetched_plots)}')

    new_scenes = results['scenes'] + fetched_plots
    
    # Index each scene's first and last sentence in the whole book content and extract text
    book_content = book['content']
    for scene in new_scenes:
        # Match first_sentence and last_sentence independently in book_content
        matched_first = find_best_match_sentence(book_content, scene.get('first_sentence')) if scene.get('first_sentence') else None
        matched_last = find_best_match_sentence(book_content, scene.get('last_sentence')) if scene.get('last_sentence') else None

        first_sentence_index = book_content.find(matched_first) if matched_first else -1
        last_sentence_index = book_content.find(matched_last) + len(matched_last) if matched_last and book_content.find(matched_last) != -1 else -1

        scene['first_sentence_index'] = first_sentence_index
        scene['last_sentence_index'] = last_sentence_index

        # Extract text between first and last sentence if both are found
        if first_sentence_index != -1 and last_sentence_index != -1 and first_sentence_index <= last_sentence_index:
            scene['text'] = book_content[first_sentence_index:last_sentence_index]
        else:
            scene['text'] = ''
    
    # Sort all scenes by their first sentence index
    # new_scenes = sorted(new_scenes, key=lambda x: x.get('first_sentence_index', float('inf')))

    results['scenes'] = new_scenes

    # Save restored results (together with the original results)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return book['title']


def clean_scenes(book):
    """
    Clean and refine extracted scenes by optimizing the content in interactions.
    
    This function loads extracted scenes, uses an LLM to refine the content
    in each interaction (removing redundancy while preserving meaning), and saves
    the cleaned results to the 'cleaned' folder.

    Args:
        book (dict): Book data containing title, author, and content

    Returns:
        str: Book title
    """
    from utils import set_cache_path, extract_json
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')

    save_dir = f'{args.output_dir}/cleaned'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{book["title"]}.json'

    if os.path.exists(save_path) and not args.regenerate:
        return book['title']

    logger.info(f"Cleaning scenes for book: {book['title']}")

    # Load extracted scene data
    with open(f'{args.output_dir}/extracted/{book["title"]}.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    scenes = results['scenes']

    # Detect language from first scene text
    if len(scenes) > 0:
        from utils import lang_detect
        language = lang_detect(scenes[0].get('text', '')[:100])
        language = {'zh': 'Chinese', 'en': 'English'}.get(language, 'English')
    else:
        language = 'English'

    # Clean interactions for each scene
    for scene in tqdm(scenes, desc=f"Cleaning {book['title']}", leave=False):
        if scene is None or not scene.get('interactions'):
            continue

        interactions = scene.get('interactions', [])
        key_characters = scene.get('key_characters', [])
        scenario = scene.get('scenario', '')
        summary = scene.get('summary', '')

        prompt = f"""You are refining character interactions extracted from a book scene.

## INTERACTION FORMAT
**[thought] speech (action)**
- **[thought]**: Internal perspective, emotions, motivations that guide the character's upcoming speech/action (REQUIRED; every interaction MUST start with [thought]; can reappear mid-turn if a new thought arises)
- **speech**: Exact spoken words (optional, can repeat)
- **(action)**: Body language, facial expressions, tone, gestures, physical actions (optional, can repeat) — do NOT put internal thoughts here

## TASK
Refine the entire `content` of each interaction (thought, speech, and action). The current interactions may have redundant thoughts, unnatural speech, or awkward actions. Make each interaction **concise, emotionally authentic, and natural**.

**Rules:**
1. **[thought]**: Remove thoughts that merely repeat or paraphrase the speech/action that follows; remove over-interpretation or excessive psychological analysis; keep thoughts that reveal genuine internal conflict, hidden feelings, or motivations not obvious from speech/action; if a thought adds no value, replace it with a brief, authentic inner voice enclosed in `[]`
2. **speech**: Do NOT add, remove, or rephrase any words — only fix perspective if third-person pronouns refer to the character(s) in `characters`
3. **(action)**: Do NOT add, remove, or rephrase any words — only fix perspective if third-person pronouns refer to the character(s) in `characters`
4. Do NOT change the `characters` field
5. If any part of `content` (thought/speech/action) uses third-person pronouns (he/she/they/his/her/their/etc.) to refer to the character(s) in `characters`, **rewrite only that pronoun/phrase** from the character's own first-person perspective or using their name directly — do not change anything else; if `characters` contains multiple people and a shared first-person perspective is unnatural, an observer's perspective is acceptable, but you MUST refer to them by the names listed in `characters` — never use pronouns

## OUTPUT FORMAT (JSON)
{{
    "interactions": [
        {{
            "characters": ["name"],
            "content": "[refined thought] speech (action) ..."
        }}
    ]
}}

## REQUIREMENTS
1. Output MUST be valid JSON
2. Output MUST contain exactly {len(interactions)} interactions (same as input — do not merge, split, or remove any)
3. Do NOT change the `characters` field of any interaction
4. Output in {language}

## SCENE CONTEXT
Book: {book['title']}
Summary: {summary}
Scenario: {scenario}
Key Characters: {json.dumps([c.get('name') for c in key_characters], ensure_ascii=False)}

## INTERACTIONS TO CLEAN
{json.dumps(interactions, ensure_ascii=False, indent=2)}
"""

        scene_key_char_names = set(c.get('name') for c in key_characters)

        original_characters_list = [
            inter['characters'] if isinstance(inter.get('characters'), list) else [inter.get('characters')]
            for inter in interactions
        ]

        def parse_response(response, expected_count, scene_key_char_names, **kwargs):
            try:
                if 'interactions' not in response:
                    return False
                cleaned = response['interactions']
                if len(cleaned) != expected_count:
                    logger.warning(f"Interaction count mismatch: expected {expected_count}, got {len(cleaned)}")
                    return False
                # Validate each interaction has required fields
                for i, inter in enumerate(cleaned):
                    if 'characters' not in inter or 'content' not in inter:
                        return False
                    # Verify all interaction characters are in scene's key_characters
                    chars = inter['characters'] if isinstance(inter['characters'], list) else [inter['characters']]
                    missing_chars = {c for c in chars if c != 'Environment'} - scene_key_char_names
                    if missing_chars:
                        logger.warning(f"Interaction characters not in key_characters: {missing_chars}, rejecting response")
                        return False
                    # Verify characters match the original interaction
                    orig_chars = original_characters_list[i]
                    new_chars = inter['characters'] if isinstance(inter['characters'], list) else [inter['characters']]
                    if sorted(orig_chars) != sorted(new_chars):
                        logger.warning(f"Interaction {i} characters mismatch: expected {orig_chars}, got {new_chars}, rejecting response")
                        return False
                return response
            except Exception as e:
                logger.error(f"Error parsing clean_scenes response: {e}")
                return False

        response = get_response_json(
            [extract_json, parse_response],
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            expected_count=len(interactions),
            scene_key_char_names=scene_key_char_names,
            max_retry=3
        )

        if response and 'interactions' in response:
            scene['interactions'] = response['interactions']
        else:
            logger.warning(f"Failed to clean interactions for scene: {scene.get('summary', '')[:50]}")

    results['scenes'] = scenes

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Cleaned scenes saved to {save_path}")
    return book['title']


def standardize_character_names(book):
    """
    Standardize character names across all scenes of a book.

    This function:
    1. Collects all key characters from every scene (name + description, deduplicated by name)
    2. Sends them to an LLM in batches of 50 to identify aliases and determine official names
    3. Builds a unified official character list with aliases
    4. Replaces all character name references in scenes (key_characters + interactions) with official names
    5. Saves the result to the 'standardized' folder

    Args:
        book (dict): Book data containing title, author, and content

    Returns:
        str: Book title
    """
    from utils import set_cache_path, extract_json
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')

    save_dir = f'{args.output_dir}/standardized'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{book["title"]}.json'

    if os.path.exists(save_path) and not args.regenerate:
        return book['title']

    logger.info(f"Standardizing character names for book: {book['title']}")

    # Load cleaned scene data
    with open(f'{args.output_dir}/cleaned/{book["title"]}.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    scenes = results['scenes']

    # ------------------------------------------------------------------ #
    # Step 1: Collect all unique characters (by name, first-occurrence    #
    #         description wins)                                            #
    # ------------------------------------------------------------------ #
    seen_names = {}  # name -> description
    for scene in scenes:
        for char in scene.get('key_characters', []):
            name = char.get('name', '').strip()
            if not name or name == 'Environment':
                continue
            if name not in seen_names:
                seen_names[name] = char.get('description', '')

    # Build ordered list: [{"name": ..., "description": ...}, ...]
    all_characters = [{'name': n, 'description': d} for n, d in seen_names.items()]
    all_input_names = set(seen_names.keys())

    logger.info(f"Total unique character names collected: {len(all_characters)}")

    # ------------------------------------------------------------------ #
    # Step 2: Send to LLM in batches of 20, accumulating official list    #
    # ------------------------------------------------------------------ #
    BATCH_SIZE = 20

    official_character_list = []  # grows across batches
    failed = False
    fail_prompts = []

    for batch_start in range(0, len(all_characters), BATCH_SIZE):
        batch = all_characters[batch_start: batch_start + BATCH_SIZE]
        batch_names = {c['name'] for c in batch}

        # Build the "existing official characters" section (empty on first batch)
        if official_character_list:
            existing_section = json.dumps(official_character_list, ensure_ascii=False, indent=2)
        else:
            existing_section = "[]"

        prompt = f"""You are helping to standardize character names in the book "{book['title']}".

## TASK
You will be given:
1. A list of **existing official characters** (already processed in previous batches) — may be empty on the first batch.
2. A list of **new characters** to process in this batch.

Your job is to produce an updated **official character list** by processing every new character in the batch.

## PRIORITY RULES (in order)

### PRIORITY 1 — Cover every character (MANDATORY, no exceptions)
Every single name from the new batch AND every name/alias from the existing official list MUST appear in the output — either as a `name` or inside `alias`. **Never drop any name.** Even minor characters with no more formal name must be kept as their own entry with their original name as the official `name`.

### PRIORITY 2 — Choose the most formal name (best effort)
Once coverage is guaranteed, try to use the most formal and complete name as the official `name`:
- For each new character, check if it clearly refers to the same person as an existing official character. If yes, merge them (add the new name as an alias, or upgrade the official name if the new name is more formal).
- If a new character's name is MORE formal/complete than an existing official character's name (e.g., existing entry uses nickname "Lizzy" but new batch contains full name "Elizabeth Bennet"), **update** the official `name` to the more formal version and move the old name into `alias`.
- If the existing official name is already the most formal, keep it unchanged.
- If a character has no more formal name available (e.g., a minor character only referred to as "the innkeeper" or "Old Pete"), simply keep their original name as the official `name` — do NOT invent a more formal name.
- Collect all alternative names / nicknames / short forms as `alias`.

## OUTPUT FORMAT (JSON)
{{
    "character_list": [
        {{
            "name": "Official name (most formal if known, otherwise original name)",
            "description": "Brief character description (~20 words)",
            "alias": ["alternative name 1", "alternative name 2", ...]
        }}
    ]
}}

## REQUIREMENTS
1. Output MUST be valid JSON following the format above.
2. Every entry MUST have `name`, `description`, and `alias` fields (`alias` can be an empty list).
3. No duplicate entries — each real-world character appears exactly once.
4. The `alias` list must NOT include the official `name` itself.
5. **Coverage check (enforced)**: every name from the existing official list AND every name from the new batch MUST appear as either a `name` or in an `alias` in the output. This is non-negotiable.
6. **Case sensitivity**: name matching in downstream code is exact (case-sensitive). If the same character appears under different capitalizations (e.g., "elizabeth" vs "Elizabeth" vs "ELIZABETH"), ALL variants MUST be listed — the most formal/canonical casing as the official `name`, and every other casing variant as a separate entry in `alias`. Do NOT silently drop any casing variant.

## EXISTING OFFICIAL CHARACTERS (from previous batches)
{existing_section}

## NEW CHARACTERS TO PROCESS (this batch)
{json.dumps(batch, ensure_ascii=False, indent=2)}
"""

        # Collect all names that must be covered in the output
        existing_names_to_cover = set()
        for entry in official_character_list:
            existing_names_to_cover.add(entry['name'])
            existing_names_to_cover.update(entry.get('alias', []))
        required_names = existing_names_to_cover | batch_names

        def parse_response(response, required_names, **kwargs):
            try:
                if 'character_list' not in response:
                    logger.warning("Response missing 'character_list' key")
                    return False
                char_list = response['character_list']
                if not isinstance(char_list, list):
                    logger.warning("'character_list' is not a list")
                    return False
                # Validate each entry has required fields
                for entry in char_list:
                    if 'name' not in entry or 'alias' not in entry or 'description' not in entry:
                        logger.warning(f"Entry missing required fields: {entry}")
                        return False
                # Coverage check: all required names must appear as name or alias
                covered = set()
                for entry in char_list:
                    covered.add(entry['name'])
                    covered.update(entry.get('alias', []))
                missing = required_names - covered
                if missing:
                    # Try case-insensitive fallback before failing
                    lower_to_entry = {}
                    for entry in char_list:
                        lower_to_entry[entry['name'].lower()] = entry
                        for alias in entry.get('alias', []):
                            lower_to_entry[alias.lower()] = entry
                    still_missing = set()
                    for raw_name in missing:
                        matched_entry = lower_to_entry.get(raw_name.lower())
                        if matched_entry is not None:
                            # Add the original casing variant as an alias
                            if raw_name not in matched_entry['alias'] and raw_name != matched_entry['name']:
                                matched_entry['alias'].append(raw_name)
                            covered.add(raw_name)
                            logger.info(f"Case-insensitive fallback in parse_response: '{raw_name}' -> '{matched_entry['name']}', added to alias")
                        else:
                            still_missing.add(raw_name)
                    if still_missing:
                        logger.warning(f"Coverage check failed — missing names: {still_missing}")
                        return False
                return response
            except Exception as e:
                logger.error(f"Error in parse_response for standardize_character_names: {e}")
                return False

        response = get_response_json(
            [extract_json, parse_response],
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            required_names=required_names,
            max_retry=5
        )

        if response and 'character_list' in response:
            official_character_list = response['character_list']
            logger.info(f"Batch {batch_start // BATCH_SIZE + 1}: official list now has {len(official_character_list)} entries")
        else:
            logger.error(f"Failed to standardize character names for batch starting at {batch_start}")
            fail_prompts.append(prompt)
            failed = True
            break

    # ------------------------------------------------------------------ #
    # Step 3: If any batch failed, record and return early                #
    # ------------------------------------------------------------------ #
    if failed:
        results.setdefault('fail', [])
        for p in fail_prompts:
            results['fail'].append({'standardize_character_names_prompt': p})
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.warning(f"Standardization failed for book: {book['title']}, partial results saved.")
        return book['title']

    # ------------------------------------------------------------------ #
    # Step 4: Build lookup table: any name/alias -> official name         #
    # ------------------------------------------------------------------ #
    name_to_official = {}  # raw name -> official name
    for entry in official_character_list:
        official = entry['name']
        name_to_official[official] = official
        for alias in entry.get('alias', []):
            name_to_official[alias] = official

    # Warn about any input names not covered (should not happen after parse_response checks)
    uncovered = all_input_names - set(name_to_official.keys())
    if uncovered:
        logger.warning(f"The following names were not covered by the official list: {uncovered}")

    # ------------------------------------------------------------------ #
    # Step 5: Replace names in all scenes                                 #
    # ------------------------------------------------------------------ #
    for scene in scenes:
        # --- key_characters ---
        new_key_characters = []
        seen_official_in_scene = set()
        for char in scene.get('key_characters', []):
            raw_name = char.get('name', '').strip()
            official = name_to_official.get(raw_name, raw_name)
            if official in seen_official_in_scene:
                # Duplicate after merging — skip
                continue
            seen_official_in_scene.add(official)
            char['name'] = official
            new_key_characters.append(char)
        scene['key_characters'] = new_key_characters

        # --- interactions ---
        for interaction in scene.get('interactions', []):
            if 'characters' in interaction:
                chars = interaction['characters']
                if isinstance(chars, list):
                    interaction['characters'] = [
                        name_to_official.get(c, c) for c in chars
                    ]
                else:
                    interaction['characters'] = name_to_official.get(chars, chars)

    # ------------------------------------------------------------------ #
    # Step 6: Save results                                                #
    # ------------------------------------------------------------------ #
    results['scenes'] = scenes

    # Insert character_list before fail field
    ordered_results = {}
    for k, v in results.items():
        if k == 'fail':
            ordered_results['character_list'] = official_character_list
        ordered_results[k] = v
    # In case 'fail' key doesn't exist
    if 'character_list' not in ordered_results:
        ordered_results['character_list'] = official_character_list

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(ordered_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Standardized character names saved to {save_path}")
    return book['title']


def merge_interactions_for_book(book):
    """
    Merge consecutive interactions with identical character lists within each scene.
    Reads from {output_dir}/standardized/, writes to {output_dir}/standardized_merge/.

    Args:
        book (dict): Book data containing title

    Returns:
        str: Book title
    """
    from utils import set_cache_path
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')

    save_dir = f'{args.output_dir}/standardized_merge'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{book["title"]}.json'

    if os.path.exists(save_path) and not args.regenerate:
        return book['title']

    input_path = f'{args.output_dir}/standardized/{book["title"]}.json'
    if not os.path.exists(input_path):
        logger.warning(f"Standardized data not found for {book['title']}, skipping merge_interactions.")
        return book['title']

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scenes = data.get('scenes', [])
    total_before = 0
    total_after = 0

    for scene in scenes:
        interactions = scene.get('interactions', [])
        total_before += len(interactions)

        if not interactions:
            continue

        merged = []
        current = {
            'characters': interactions[0]['characters'],
            'content': interactions[0]['content'],
        }
        for interaction in interactions[1:]:
            if interaction['characters'] == current['characters']:
                current['content'] = current['content'] + ' ' + interaction['content']
            else:
                merged.append(current)
                current = {
                    'characters': interaction['characters'],
                    'content': interaction['content'],
                }
        merged.append(current)
        scene['interactions'] = merged
        total_after += len(merged)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[{book['title']}] merge_interactions: {total_before} -> {total_after} "
        f"(merged {total_before - total_after})"
    )
    return book['title']


def clean_duplicate_scenes_for_book(book):
    """
    Detect and remove duplicate scenes using LLM.
    Reads from {output_dir}/standardized_merge/, overwrites in place.

    Args:
        book (dict): Book data containing title

    Returns:
        str: Book title
    """
    import copy
    from utils import set_cache_path
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')

    sm_path = f'{args.output_dir}/standardized_merge/{book["title"]}.json'
    if not os.path.exists(sm_path):
        logger.warning(f"standardized_merge file not found for {book['title']}, skipping clean_duplicates.")
        return book['title']

    with open(sm_path, 'r', encoding='utf-8') as f:
        sm_data = json.load(f)

    scenes = sm_data.get('scenes', [])
    if not scenes:
        logger.info(f"[{book['title']}] No scenes found, skipping clean_duplicates.")
        return book['title']

    # ── LLM duplicate detection ───────────────────────────────────────────────
    DUPLICATE_DETECTION_PROMPT = """\
You are given a list of scenes from a book. Each scene has an index and a summary.
Your task is to identify pairs of scenes that describe the SAME narrative event (i.e., duplicate scenes).

Two scenes are duplicates if they cover the same plot event, even if the wording differs.
For each duplicate pair, decide which scene contains MORE information (richer summary, more detail).
The scene with LESS information should be deleted.

Return a JSON array of objects. Each object represents one duplicate pair to resolve:
{{
  "keep_index": <scene index to keep (the richer one)>,
  "delete_index": <scene index to delete (the less informative one)>,
  "reason": "<brief reason>"
}}

If there are no duplicates, return an empty array: []

IMPORTANT:
- Only flag scenes that are clearly duplicates of the same narrative event.
- Do NOT flag scenes that are merely similar in theme but cover different moments.
- Return ONLY valid JSON, no extra text.

Scenes:
{scenes_json}
"""

    def _build_compact(scenes_list):
        return [{'index': i, 'summary': s.get('summary', '')} for i, s in enumerate(scenes_list)]

    def _call_llm_for_duplicates(compact_scenes, group_idx, total_scenes):
        scenes_json = json.dumps(compact_scenes, ensure_ascii=False, indent=2)
        prompt = DUPLICATE_DETECTION_PROMPT.format(scenes_json=scenes_json)
        messages = [{'role': 'user', 'content': prompt}]
        logger.info(
            f"[{book['title']}] Group {group_idx}: sending {len(compact_scenes)} scenes "
            f"(indices {compact_scenes[0]['index']}–{compact_scenes[-1]['index']}) to LLM..."
        )
        response = get_response(model=args.model, messages=messages)
        if not response:
            logger.error(f"[{book['title']}] Group {group_idx}: LLM returned empty response.")
            return []
        if response.strip() in ('[]', '[ ]'):
            return []
        result = extract_json(response, model=args.model)
        if result is None:
            stripped = response.strip()
            stripped = re.sub(r'^```[a-z]*\n?', '', stripped, flags=re.IGNORECASE)
            stripped = re.sub(r'\n?```$', '', stripped).strip()
            if stripped in ('[]', '[ ]'):
                return []
            logger.error(f"[{book['title']}] Group {group_idx}: Failed to parse LLM response: {response[:500]}")
            return []
        if not isinstance(result, list):
            logger.error(f"[{book['title']}] Group {group_idx}: LLM returned non-list result: {result}")
            return []
        valid = []
        for entry in result:
            if (
                isinstance(entry, dict)
                and 'keep_index' in entry
                and 'delete_index' in entry
                and isinstance(entry['keep_index'], int)
                and isinstance(entry['delete_index'], int)
                and 0 <= entry['keep_index'] < total_scenes
                and 0 <= entry['delete_index'] < total_scenes
                and entry['keep_index'] != entry['delete_index']
            ):
                valid.append(entry)
            else:
                logger.warning(f"[{book['title']}] Group {group_idx}: Invalid entry skipped: {entry}")
        return valid

    GROUP_SIZE = 20
    total = len(scenes)
    compact_all = _build_compact(scenes)
    all_pairs = []
    seen_pairs = set()
    group_idx = 0
    start = 0
    num_groups = (total + GROUP_SIZE - 1) // GROUP_SIZE
    with tqdm(total=num_groups, desc=f"Clean duplicates {book['title']}", leave=False) as pbar:
        while start < total:
            end = min(start + GROUP_SIZE, total)
            overlap_start = max(0, start - 1)
            group_compact = compact_all[overlap_start:end]
            pairs = _call_llm_for_duplicates(group_compact, group_idx, total)
            for pair in pairs:
                key = (min(pair['keep_index'], pair['delete_index']),
                       max(pair['keep_index'], pair['delete_index']))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    all_pairs.append(pair)
            group_idx += 1
            start = end
            pbar.update(1)

    logger.info(f"[{book['title']}] LLM found {len(all_pairs)} duplicate pair(s) across {group_idx} group(s).")

    if not all_pairs:
        logger.info(f"[{book['title']}] No duplicates found.")
        # Still save a clean copy to standardized_merge_clean/
        final_dir = os.path.join(args.output_dir, 'standardized_merge_clean')
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, f'{book["title"]}.json')
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(sm_data, f, ensure_ascii=False, indent=2)
        return book['title']

    # ── remove duplicates ─────────────────────────────────────────────────────
    deleted_indices = set()
    for pair in all_pairs:
        del_idx = pair['delete_index']
        if del_idx not in deleted_indices:
            deleted_indices.add(del_idx)

    new_scenes = []
    for old_idx, scene in enumerate(scenes):
        if old_idx not in deleted_indices:
            new_scenes.append(copy.deepcopy(scene))

    new_sm_data = copy.deepcopy(sm_data)
    new_sm_data['scenes'] = new_scenes

    # Save to standardized_merge_clean/ (same as clean.py)
    final_dir = os.path.join(args.output_dir, 'standardized_merge_clean')
    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, f'{book["title"]}.json')
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(new_sm_data, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[{book['title']}] clean_duplicate_scenes: deleted {len(deleted_indices)} scene(s): "
        f"{sorted(deleted_indices)}, saved to {final_path}"
    )
    return book['title']


def build_profiles_initialization(book):
    """
    Build initial character profiles for all characters before any scene begins.

    This function generates a "starting state" profile for each character — what they
    are like BEFORE the events of the book unfold. The profile must NOT spoil any
    specific plot events; it should only capture the character's initial state
    (background, personality, relationships, etc.) as inferred from the full story.

    The model is given scene summaries, all interactions, and the character's
    key_character entry for each relevant scene, but is instructed to describe
    only the character's state at the very beginning of the story.

    Args:
        book (dict): Book data containing title, author, and content

    Returns:
        str: Book title
    """
    from utils import set_cache_path, extract_json
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')

    save_dir = f'{args.output_dir}/character_profiles_initialization'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{book["title"]}.json'

    if os.path.exists(save_path) and not args.regenerate:
        return book['title']

    logger.info(f"Building initial character profiles for book: {book['title']}")

    # Load standardized scene data (contains character_list and standardized scenes)
    standardized_path = f'{args.output_dir}/standardized_merge_clean/{book["title"]}.json'
    if not os.path.exists(standardized_path):
        logger.warning(f"Standardized data not found for {book['title']}, skipping.")
        return book['title']

    with open(standardized_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    scenes = results.get('scenes', [])
    official_character_list = results.get('character_list', [])

    if not scenes or not official_character_list:
        logger.warning(f"No scenes or character list found for {book['title']}, skipping.")
        return book['title']

    # Detect language from first scene text
    if len(scenes) > 0:
        from utils import lang_detect
        language = lang_detect(scenes[0].get('text', '')[:100])
        language = {'zh': 'Chinese', 'en': 'English'}.get(language, 'English')
    else:
        language = 'English'

    # Build a lookup: official name -> set of aliases
    official_names = [entry['name'] for entry in official_character_list]

    # For each official character, collect relevant scene data
    # We index scenes by official character name
    # character -> list of (scene_index, scene)
    character_scene_map = {name: [] for name in official_names}

    for i_s, scene in enumerate(scenes):
        if scene is None:
            continue
        scene_char_names = {c['name'] for c in scene.get('key_characters', [])}
        for name in official_names:
            if name in scene_char_names:
                character_scene_map[name].append(i_s)

    # Token budget per character prompt (leave room for system overhead)
    MAX_SCENE_TOKENS = 100000  # max tokens for scene data section in the prompt

    # Reference dimensions for the model (model decides which to use)
    dimension_hints = (
        "Physical Description, Social Standing, Professional Identity, "
        "Core Personality, Mental Health Status, Cognitive Biases, Moral Code, "
        "Speech Patterns, Signature Catchphrases, Core Motivations, Core Fears, "
        "Skills & Expertise, Supernatural Powers, Wealth & Assets, "
        "Faction Loyalty, Historical Baggage, Key Relationships, "
        "Emotional Debts, Backstory Milestones"
    )

    profile_prompt_template = """\
You are building an **initial character profile** — a snapshot of who {character_name} is at the very START of the story, BEFORE any of the depicted events unfold.

## TASK
Based on the scene data provided below (summaries, interactions, and the character's role in each scene), infer and describe this character's **initial state** at the beginning of the book.

## CRITICAL RULES
1. **NO SPOILERS**: Do NOT reveal specific plot events, outcomes, deaths, betrayals, or any concrete story developments. Describe only the character's baseline state (personality, background, relationships, etc.) as it would exist before the story begins. If a relationship, trait, belief, or any other characteristic only emerges or is established during a specific scene in the story, do NOT include it in this profile.
2. **Infer backwards**: Use what happens in the story to infer what the character must have been like at the start — their traits, history, relationships — without narrating the events themselves.
3. **Select relevant dimensions**: Choose only the dimensions that are meaningful for this character. You are NOT required to cover all dimensions. Suggested dimensions for reference (use, skip, or add your own as appropriate):
   {dimension_hints}
4. **Format**: Output a structured profile with clearly labeled dimensions. Be concise: avoid filler phrases, redundant elaboration, or vague generalities — every sentence should carry specific, meaningful information. Do NOT include any preamble or meta-commentary (e.g., "This profile describes X at the beginning of the story...") — start directly with the first dimension.
5. **Language**: Output in {language}.
6. **Grounding**: Base the profile on the provided scene data and/or your existing knowledge of the character. Do NOT fabricate details.

## BOOK & CHARACTER
Book: {book_title}
Character: {character_name}

## SCENE DATA (for reference — do NOT narrate these events in the profile)
{scene_data}

Now generate the initial character profile, starting with ===Profile===.
"""

    profiles = {}

    for character_name in tqdm(official_names, desc=f"Building init profiles {book['title']}", leave=False):
        involved_scene_indices = character_scene_map[character_name]

        if not involved_scene_indices:
            logger.info(f"No scenes found for character: {character_name}, skipping.")
            profiles[character_name] = ''
            continue

        # Build scene data list, respecting token budget
        scene_data_list = []
        total_tokens = 0

        for i_s in involved_scene_indices:
            scene = scenes[i_s]
            if scene is None:
                continue

            # Find this character's key_character entry
            key_char_info = None
            for kc in scene.get('key_characters', []):
                if kc.get('name') == character_name:
                    key_char_info = kc
                    break

            scene_entry = {
                'scene_summary': scene.get('summary', ''),
                'character_info': key_char_info,
                'interactions': scene.get('interactions', []),
            }

            entry_tokens = len(encode(json.dumps(scene_entry, ensure_ascii=False)))

            if total_tokens + entry_tokens > MAX_SCENE_TOKENS:
                # If we already have some scenes, stop adding more
                if scene_data_list:
                    logger.debug(
                        f"Token budget reached for {character_name} at scene {i_s}, "
                        f"using {len(scene_data_list)}/{len(involved_scene_indices)} scenes."
                    )
                    break
                else:
                    # First scene already exceeds budget — truncate interactions
                    # Keep summary + key_char_info, truncate interactions list
                    budget_for_interactions = MAX_SCENE_TOKENS - len(
                        encode(json.dumps({
                            'scene_summary': scene_entry['scene_summary'],
                            'character_info': key_char_info,
                            'interactions': [],
                        }, ensure_ascii=False))
                    )
                    truncated_interactions = []
                    inter_tokens = 0
                    for inter in scene_entry['interactions']:
                        t = len(encode(json.dumps(inter, ensure_ascii=False)))
                        if inter_tokens + t > budget_for_interactions:
                            break
                        truncated_interactions.append(inter)
                        inter_tokens += t
                    scene_entry['interactions'] = truncated_interactions
                    scene_data_list.append(scene_entry)
                    total_tokens += len(encode(json.dumps(scene_entry, ensure_ascii=False)))
                    break

            scene_data_list.append(scene_entry)
            total_tokens += entry_tokens

        scene_data_str = json.dumps(scene_data_list, ensure_ascii=False, indent=2)

        character_prompt = profile_prompt_template.format(
            character_name=character_name,
            book_title=book['title'],
            dimension_hints=dimension_hints,
            language=language,
            scene_data=scene_data_str,
        )

        logger.debug(f"Initial profile prompt for {character_name} ({total_tokens} scene tokens):\n{character_prompt[:300]}...")

        # Get profile from LLM with retries
        nth_generation = 0
        profile = ''
        while True:
            kwargs = dict(
                model=args.model,
                messages=[{"role": "user", "content": character_prompt}],
            )
            if nth_generation > 0:
                kwargs['nth_generation'] = nth_generation

            raw = get_response(**kwargs)

            try:
                if "===Profile===" in raw:
                    profile = raw.split("===Profile===", 1)[1].strip()
                else:
                    profile = raw.strip()
                if profile.startswith('I apologize'):
                    nth_generation += 1
                    if nth_generation > 5:
                        logger.warning(f"Failed to generate initial profile for {character_name} after 5 retries (model apologized). Last response: {raw}")
                        profile = ''
                        break
                    continue
                break
            except Exception:
                nth_generation += 1
                if nth_generation > 5:
                    logger.warning(f"Failed to generate initial profile for {character_name} after 5 retries. Last response: {raw}")
                    profile = ''
                    break
                continue

        profiles[character_name] = profile

    # Save profiles (only the profile text, not scene data)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    logger.info(f"Initial character profiles saved to {save_path}")
    return book['title']


def build_profiles_dynamic(book):
    """
    Dynamically build and update character profiles as the story progresses scene by scene.

    For each character, this function:
    1. Starts from the initial profile (from character_profiles_initialization).
    2. After each scene the character appears in, uses an LLM to:
       a. Decide whether the character profile needs updating (using the next scene as a
          lookahead reference to detect gradual changes that span multiple scenes).
       b. Maintain a hidden_tracker that records events/signals that may lead to future
          profile changes, even if the profile itself hasn't changed yet.
       c. Update the profile only when meaningful changes are detected.
    3. Before the first scene, generates an initial short description of the character
       based solely on their initial profile, summarizing their identity and goals at the
       very start of the story. This description is stored in profile_history[0].
    4. After every scene the character appears in (regardless of profile update), generates
       a short description summarizing the character's current identity and immediate goals
       (using the next scene as reference for what they're about to do).

    Output is saved to dataset/extracted_data/character_dynamic/{book_title}.json with structure:
    {
        "character_name": {
            "profile_history": [
                {
                    "scene_index": int,       // scene index after which profile was updated
                    "profile": str,          // updated profile text
                    "description": str       // (only in entry [0]) initial short description generated before any scenes
                }
            ],
            "scene_descriptions": [
                {
                    "scene_index": int,       // scene index
                    "enhanced_motivation": str, // enhanced motivation for THIS scene (generated from the scene's lookahead, or from a dedicated first-scene call)
                    "description": str,        // short description: identity + immediate goals (generated at end of this scene)
                    "hidden_tracker": str     // tracker of potential future changes
                }
            ]
        }
    }

    Args:
        book (dict): Book data containing title, author, and content

    Returns:
        str: Book title
    """
    from utils import set_cache_path, extract_json
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')

    save_dir = os.path.join('dataset', 'extracted_data', 'character_dynamic')
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{book["title"]}.json'

    if os.path.exists(save_path) and not args.regenerate:
        return book['title']

    logger.info(f"Building dynamic character profiles for book: {book['title']}")

    # ------------------------------------------------------------------ #
    # Step 1: Load standardized scenes and initial profiles               #
    # ------------------------------------------------------------------ #
    standardized_path = f'{args.output_dir}/standardized_merge_clean/{book["title"]}.json'
    if not os.path.exists(standardized_path):
        logger.warning(f"Standardized data not found for {book['title']}, skipping.")
        return book['title']

    init_profiles_path = f'{args.output_dir}/character_profiles_initialization/{book["title"]}.json'
    if not os.path.exists(init_profiles_path):
        logger.warning(f"Initial profiles not found for {book['title']}, skipping.")
        return book['title']

    with open(standardized_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    with open(init_profiles_path, 'r', encoding='utf-8') as f:
        init_profiles = json.load(f)

    scenes = results.get('scenes', [])
    official_character_list = results.get('character_list', [])

    if not scenes or not official_character_list:
        logger.warning(f"No scenes or character list found for {book['title']}, skipping.")
        return book['title']

    # Detect language from first scene text
    if len(scenes) > 0:
        from utils import lang_detect
        language = lang_detect(scenes[0].get('text', '')[:100])
        language = {'zh': 'Chinese', 'en': 'English'}.get(language, 'English')
    else:
        language = 'English'

    official_names = [entry['name'] for entry in official_character_list]

    # ------------------------------------------------------------------ #
    # Step 2: Build per-character scene index list                        #
    # ------------------------------------------------------------------ #
    character_scene_map = {name: [] for name in official_names}
    for i_s, scene in enumerate(scenes):
        if scene is None:
            continue
        scene_char_names = {c['name'] for c in scene.get('key_characters', [])}
        for name in official_names:
            if name in scene_char_names:
                character_scene_map[name].append(i_s)


    # ------------------------------------------------------------------ #
    # Step 3: Prompt templates                                            #
    # ------------------------------------------------------------------ #

    # Prompt for generating a short description from the initial profile
    # (before any scene has been processed)
    initial_description_prompt_template = """\
You are preparing a character introduction for a story simulation system. Based solely on the character's initial profile, write a short description that will be used alongside other characters' descriptions to understand who this character is at the very start of the story.

## CHARACTER
**Name:** {character_name}

**Initial Profile** (who {character_name} is at the very start of the story):
{current_profile}

## YOUR TASK
Write a short description (50–80 words) of {character_name} AS OF THE START OF THE STORY (before any scenes have occurred). Cover:
1. **Identity**: Who is {character_name}? (role, key relationships, current situation at the story's opening)
2. **Immediate goals/intentions**: What does {character_name} want or plan to do at the start of the story?
Be specific and concrete. Write in third person.

## OUTPUT FORMAT (JSON)
{{
    "description": "Short description text (50–80 words, third person, initial identity + initial goals)"
}}

## REQUIREMENTS
1. Output MUST be valid JSON.
2. Output in {language}.
"""

    # Prompt for enhancing the motivation of the FIRST scene a character appears in
    # (no previous scene exists, so we use the initial profile + first scene content)
    first_scene_enhance_prompt_template = """\
You are preparing a character for their first scene in a story. Thus, actors can have a better understanding of the character's background and motivations to act out the scene.

## CHARACTER
**Name:** {character_name}

**Initial Profile** (who {character_name} is at the very start of the story):
{current_profile}

## FIRST SCENE WHERE {character_name} APPEARS (Scene #{scene_index})
**Summary:** {scene_summary}
**Scenario:** {scene_scenario}
**Character's motivation in this scene:**
{character_info}
**Interactions in this scene:**
{scene_interactions}

## YOUR TASK
Based on {character_name}'s initial profile and the specific content of their first scene (interactions, scenario, etc.), write an enhanced motivation for {character_name} at the START of this scene. The enhanced motivation should:
- Be grounded in the character's established personality, goals, and backstory from the initial profile
- Naturally lead to and explain the character's actions in this scene
- Feel psychologically authentic and specific to this character
- Be concise (1–3 sentences)
IMPORTANT: The enhanced motivation must NOT reveal, reference, or hint at any specific events or outcomes from the scene. It should only describe the character's internal drive entering the scene, as if the scene has not yet happened.

## OUTPUT FORMAT (JSON)
{{
    "enhanced_motivation": "Enhanced motivation for {character_name} at the start of Scene #{scene_index} (1–3 sentences)"
}}

## REQUIREMENTS
1. Output MUST be valid JSON.
2. Output in {language}.
"""

    combined_prompt_template = """\
You are tracking how a character evolves throughout a story, scene by scene. Your goal is to build a comprehensive foundation (dynamic profile, hidden tracker, description, motivation...) for dramatic performance — providing the necessary background for actors to act out each scene.

## CHARACTER
**Name:** {character_name}

## CURRENT STATE (before the scene below)
**Current Profile** (snapshot of who {character_name} is RIGHT NOW, before the scene below):
{current_profile}

**Hidden Tracker** (events/signals accumulated so far that may lead to future profile changes):
{hidden_tracker}

**Brief Description from Previous Scene** (a brief introduction of who {character_name} is and what they are trying to do, generated at the end of the previous scene):
{current_description}

## SCENE JUST COMPLETED (Scene #{scene_index})
**Summary:** {scene_summary}
**Scenario:** {scene_scenario}
**Character's motivation in this scene:**
{character_info}
**Interactions in this scene:**
{scene_interactions}

## NEXT SCENE WHERE {character_name} APPEARS (lookahead reference ONLY — use ONLY to judge whether current changes are meaningful enough to update the profile now, and to infer what the character intends to do next. ALL outputs must reflect the character's state as of the END OF THE CURRENT SCENE. Do NOT reveal, reference, or hint at anything that happens in the next scene):
{next_scene_info}

## YOUR TASKS

### Task 1: Reason about dimensions
Briefly reason about which profile dimensions are STABLE (unlikely to change across the whole story) vs. DYNAMIC (can change as events unfold). Examples of stable dimensions: physical description, core fears, backstory milestones. Examples of dynamic dimensions: relationships, goals, mental health, faction loyalty, wealth.

### Task 2: Update the Hidden Tracker
Based on the scene just completed, update the hidden tracker. The tracker should record:
- Events, experiences, or interactions that signal a potential future change in the character's profile
- Accumulated emotional/psychological pressure that hasn't yet caused a visible change
- Unresolved tensions or decisions that may alter the character's trajectory
Keep the tracker concise (under 300 words). Overwrite the old tracker with the updated version.

### Task 3: Decide whether to update the profile
Decide if the character's profile should be updated NOW. Update the profile if:
- A meaningful, observable change has occurred in this scene (e.g., a relationship shift, a decision that changes their goals, a trauma that alters their personality)
- The hidden tracker shows accumulated signals that, combined with this scene, now cross a threshold for a real change
Do NOT update the profile for minor, transient reactions that don't reflect a lasting change.

### Task 4: Write a short description
Write a short description (50–80 words) of {character_name} AS OF THE END OF THE CURRENT SCENE (Scene #{scene_index}), for use in a story simulation system where it will be read alongside all other characters' descriptions to decide which scene comes next. Cover:
1. **Identity**: Who is {character_name} right now? (role, key relationships, current situation as of the end of the current scene)
2. **Immediate goals/intentions**: What does {character_name} want or plan to do next? (you may use the next scene as a reference to infer their intentions, but do NOT reveal or hint at what actually happens in the next scene)
Be specific and concrete. Write in third person. The description must be grounded in the current scene only — do NOT reveal future plot outcomes or any events from the next scene.

### Task 5: Enhance the motivation for the next scene
Based on what happened in the CURRENT SCENE and the specific content of the NEXT SCENE (interactions, scenario, etc.), write an enhanced motivation for {character_name} at the START of the next scene. The enhanced motivation should:
- Be grounded in the emotional state, decisions, and unresolved tensions from the current scene
- Capture the character's complete mental and emotional state entering the next scene: their feelings, immediate objectives, what they intend to do or say, who they plan to seek out, and what information or message they want to convey or discuss
- Naturally lead to and explain the character's specific actions and interactions in the next scene — you may hint at what the character intends to do (e.g., "plans to confront X", "intends to seek out Y to discuss Z") as long as it reads as the character's internal drive, not a spoiler of what actually happens
- Feel psychologically authentic and specific to this character
- Be concise (1–3 sentences)
IMPORTANT: The enhanced motivation must NOT reveal, reference, or hint at the actual outcomes, results, or plot developments that occur in the next scene. It should describe the character's internal drive and intentions entering the next scene, as if the next scene has not yet happened.
If there is no next scene for this character, output null.

## OUTPUT FORMAT (JSON)
{{
    "dimension_reasoning": "Brief reasoning about which dimensions are stable vs. dynamic for {character_name}",
    "hidden_tracker": "Updated tracker text, or null if there are no signals worth tracking",
    "should_update_profile": true or false,
    "updated_profile": "Full updated profile as a plain Markdown string (NOT a dict or nested object). Use the format:\\n**Dimension Name**\\nContent text\\n\\n**Another Dimension**\\nContent text\\n\\nInclude all relevant dimensions. Set to null if no update.",
    "description": "Short description text (50–80 words, third person, current identity + immediate goals)",
    "enhanced_next_motivation": "Enhanced motivation for {character_name} at the start of the next scene (1–3 sentences), or null if there is no next scene"
}}

## REQUIREMENTS
1. Output MUST be valid JSON.
2. ALL keys in the output format above MUST be present in your response — do NOT omit any key (e.g., "hidden_tracker"), even if its value is null.
3. ALL outputs (profile, description, tracker) must reflect the character's state as of the END OF THE CURRENT SCENE (Scene #{scene_index}) only.
4. The next scene is provided as a lookahead reference ONLY. Do NOT reveal, reference, or hint at any events, outcomes, or details from the next scene in any part of your output.
5. Be selective: only update the profile when there is a genuine, lasting change. Avoid over-updating.
6. `updated_profile` MUST be a plain string in Markdown format — do NOT output a JSON object or nested dict for this field.
7. Output in {language}.
"""

    # ------------------------------------------------------------------ #
    # Step 4: Process each character scene by scene                       #
    # ------------------------------------------------------------------ #
    dynamic_profiles = {}

    for character_name in tqdm(official_names, desc=f"Dynamic profiles {book['title']}", leave=False):
        involved_scene_indices = character_scene_map[character_name]

        if not involved_scene_indices:
            logger.info(f"No scenes found for character: {character_name}, skipping.")
            dynamic_profiles[character_name] = {
                'profile_history': [],
                'scene_descriptions': []
            }
            continue

        # Initialize with the initial profile
        current_profile = init_profiles.get(character_name, '')
        hidden_tracker = ''  # Empty at the start
        current_description = ''  # Description from the previous scene

        profile_history = []
        scene_descriptions = []
        enhanced_motivation_map = {}  # {scene_index: enhanced_motivation}
        fail_list = []  # Records failed prompts and responses

        # Record the initial profile as the starting state, and generate an initial short description
        if current_profile:
            initial_profile_str = dict_to_markdown(current_profile)
            initial_desc_prompt = initial_description_prompt_template.format(
                character_name=character_name,
                current_profile=initial_profile_str,
                language=language,
            )

            def parse_initial_desc_response(response, **kwargs):
                try:
                    if 'description' not in response or not response.get('description'):
                        logger.warning("Initial description response missing 'description'")
                        return False
                    return response
                except Exception as e:
                    logger.error(f"Error parsing initial description response: {e}")
                    return False

            initial_desc_response = get_response_json(
                [extract_json, parse_initial_desc_response],
                model=args.model,
                messages=[{"role": "user", "content": initial_desc_prompt}],
                max_retry=5
            )
            if not initial_desc_response:
                logger.warning(f"Failed to generate initial description for {character_name}, retrying with candidate model")
                initial_desc_response = get_response_json(
                    [extract_json, parse_initial_desc_response],
                    model=args.candidate_model,
                    messages=[{"role": "user", "content": initial_desc_prompt}],
                    max_retry=5
                )
            initial_description = initial_desc_response.get('description', '').strip() if initial_desc_response else ''
            if initial_description:
                current_description = initial_description

            profile_history.append({
                'scene_index': involved_scene_indices[0],  # first scene this character appears in
                'profile': initial_profile_str,
                'description': initial_description,
            })

        # ---- Enhance motivation for the first scene ---- #
        if involved_scene_indices:
            first_i_s = involved_scene_indices[0]
            first_scene = scenes[first_i_s]
            if first_scene is not None:
                first_key_char_info = None
                for kc in first_scene.get('key_characters', []):
                    if kc.get('name') == character_name:
                        first_key_char_info = kc.get('motivation')
                        break
                first_interactions_str = json.dumps(first_scene.get('interactions', []), ensure_ascii=False, indent=2)
                first_enhance_prompt = first_scene_enhance_prompt_template.format(
                    character_name=character_name,
                    current_profile=current_profile if current_profile else '(No profile yet)',
                    scene_index=first_i_s,
                    scene_summary=first_scene.get('summary', ''),
                    scene_scenario=first_scene.get('scenario', ''),
                    character_info=first_key_char_info if first_key_char_info else '(Not listed as key character)',
                    scene_interactions=first_interactions_str,
                    language=language,
                )

                def parse_first_enhance_response(response, **kwargs):
                    try:
                        if 'enhanced_motivation' not in response or not response.get('enhanced_motivation'):
                            logger.warning("First scene enhance response missing 'enhanced_motivation'")
                            return False
                        return response
                    except Exception as e:
                        logger.error(f"Error parsing first scene enhance response: {e}")
                        return False

                first_enhance_response = get_response_json(
                    [extract_json, parse_first_enhance_response],
                    model=args.model,
                    messages=[{"role": "user", "content": first_enhance_prompt}],
                    max_retry=5
                )
                if first_enhance_response:
                    enhanced_motivation_map[first_i_s] = first_enhance_response.get('enhanced_motivation')
                else:
                    logger.warning(f"Failed to enhance first scene motivation for {character_name} at scene {first_i_s}, retrying with candidate model")
                    first_enhance_response = get_response_json(
                        [extract_json, parse_first_enhance_response],
                        model=args.candidate_model,
                        messages=[{"role": "user", "content": first_enhance_prompt}],
                        max_retry=5
                    )
                    if first_enhance_response:
                        enhanced_motivation_map[first_i_s] = first_enhance_response.get('enhanced_motivation')
                    else:
                        logger.warning(f"Failed to enhance first scene motivation for {character_name} at scene {first_i_s} with candidate model")
                        fail_list.append({
                        'type': 'first_scene_enhance',
                        'scene_index': first_i_s,
                        'prompt': first_enhance_prompt,
                        'response': first_enhance_response,
                    })

        for idx, i_s in enumerate(involved_scene_indices):
            scene = scenes[i_s]
            if scene is None:
                continue

            # Find this character's key_character entry in the scene
            key_char_info = None
            for kc in scene.get('key_characters', []):
                if kc.get('name') == character_name:
                    key_char_info = kc.get('motivation')
                    break

            # Build scene interactions
            interactions = scene.get('interactions', [])
            interactions_str = json.dumps(interactions, ensure_ascii=False, indent=2)

            # Build next scene info (lookahead reference)
            # Look for the NEXT scene in which THIS CHARACTER appears
            next_scene_info = 'None (this character does not appear in any later scene)'
            if idx + 1 < len(involved_scene_indices):
                next_i_s = involved_scene_indices[idx + 1]
                next_scene = scenes[next_i_s]
                if next_scene is not None:
                    # Find this character's key_character entry in the next scene
                    next_key_char_info = None
                    for kc in next_scene.get('key_characters', []):
                        if kc.get('name') == character_name:
                            next_key_char_info = kc.get('motivation')
                            break

                    # Build next scene interactions
                    next_interactions = next_scene.get('interactions', [])
                    next_interactions_str = json.dumps(next_interactions, ensure_ascii=False, indent=2)

                    next_scene_info = (
                        f"Scene Index: {next_i_s}\n"
                        f"Summary: {next_scene.get('summary', '')}\n"
                        f"Scenario: {next_scene.get('scenario', '')}\n"
                        f"Character's motivation in this scene:\n{next_key_char_info if next_key_char_info else '(Not listed as key character)'}\n"
                        f"Interactions in this scene:\n{next_interactions_str}"
                    )

            # ---- Combined: profile update + description in one call ---- #
            combined_prompt = combined_prompt_template.format(
                book_title=book['title'],
                character_name=character_name,
                current_profile=current_profile if current_profile else '(No profile yet)',
                hidden_tracker=hidden_tracker if hidden_tracker else '(Empty — no signals accumulated yet)',
                scene_index=i_s,
                scene_summary=scene.get('summary', ''),
                scene_scenario=scene.get('scenario', ''),
                character_info=key_char_info if key_char_info else '(Not listed as key character)',
                scene_interactions=interactions_str,
                current_description=current_description if current_description else '(No description yet — this is the first scene)',
                next_scene_info=next_scene_info,
                language=language,
            )

            def parse_combined_response(response, **kwargs):
                try:
                    if 'should_update_profile' not in response:
                        logger.warning("Combined response missing 'should_update_profile'")
                        return False
                    if 'dimension_reasoning' not in response:
                        logger.warning("Combined response missing 'dimension_reasoning'")
                        return False
                    if 'description' not in response or not response.get('description'):
                        logger.warning("Combined response missing 'description'")
                        return False
                    if 'enhanced_next_motivation' not in response:
                        logger.warning("Combined response missing 'enhanced_next_motivation'")
                        return False
                    if response.get('should_update_profile') and not response.get('updated_profile'):
                        logger.warning("Combined response: should_update_profile=True but updated_profile is empty/null")
                        return False
                    if 'hidden_tracker' not in response:
                        response['hidden_tracker'] = hidden_tracker
                    return response
                except Exception as e:
                    logger.error(f"Error parsing combined response: {e}")
                    return False

            combined_response = get_response_json(
                [extract_json, parse_combined_response],
                model=args.model,
                messages=[{"role": "user", "content": combined_prompt}],
                max_retry=5
            )

            if combined_response:
                # Update hidden tracker
                hidden_tracker = combined_response.get('hidden_tracker', hidden_tracker)

                # Update profile if needed
                if combined_response.get('should_update_profile') and combined_response.get('updated_profile'):
                    current_profile = dict_to_markdown(combined_response['updated_profile'])
                    profile_history.append({
                        'scene_index': i_s,
                        'profile': current_profile,
                    })
                    logger.debug(f"Profile updated for {character_name} after scene {i_s}")

                # Record description
                description = combined_response.get('description', '').strip()
                enhanced_next_motivation = combined_response.get('enhanced_next_motivation')
                # Store enhanced_next_motivation into the NEXT scene's entry
                # If enhanced_next_motivation is null, fall back to the original motivation
                if idx + 1 < len(involved_scene_indices):
                    next_i_s = involved_scene_indices[idx + 1]
                    enhanced_motivation_map[next_i_s] = enhanced_next_motivation or next_key_char_info
            else:
                logger.warning(f"Failed to get combined response for {character_name} at scene {i_s}, retrying with candidate model")
                combined_response = get_response_json(
                    [extract_json, parse_combined_response],
                    model=args.candidate_model,
                    messages=[{"role": "user", "content": combined_prompt}],
                    max_retry=5
                )
                if combined_response:
                    hidden_tracker = combined_response.get('hidden_tracker', hidden_tracker)
                    if combined_response.get('should_update_profile') and combined_response.get('updated_profile'):
                        current_profile = dict_to_markdown(combined_response['updated_profile'])
                        profile_history.append({
                            'scene_index': i_s,
                            'profile': current_profile,
                        })
                    description = combined_response.get('description', '').strip()
                    enhanced_next_motivation = combined_response.get('enhanced_next_motivation')
                    if idx + 1 < len(involved_scene_indices):
                        next_i_s = involved_scene_indices[idx + 1]
                        enhanced_motivation_map[next_i_s] = enhanced_next_motivation or next_key_char_info
                else:
                    logger.warning(f"Failed to get combined response for {character_name} at scene {i_s} with candidate model")
                    description = ''
                    fail_list.append({
                    'type': 'combined',
                    'scene_index': i_s,
                    'prompt': combined_prompt,
                    'response': combined_response,
                })

            # Update current_description for the next scene
            if description:
                current_description = description

            scene_descriptions.append({
                'scene_index': i_s,
                'enhanced_motivation': enhanced_motivation_map.get(i_s),
                'description': description,
                'hidden_tracker': hidden_tracker if hidden_tracker else None,
            })

        dynamic_profiles[character_name] = {
            'profile_history': profile_history,
            'scene_descriptions': scene_descriptions,
            'fail': fail_list,
        }

    # ------------------------------------------------------------------ #
    # Step 5: Save results                                                #
    # ------------------------------------------------------------------ #
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dynamic_profiles, f, ensure_ascii=False, indent=2)

    logger.info(f"Dynamic character profiles saved to {save_path}")
    return book['title']


def enhance_scenes(book):
    """
    Enhance scene scenarios for all scenes of a book.

    Reads from standardized_merge_clean/, enhances each scene's scenario
using LLM, and saves the result to total_scenes/.

    Args:
        book (dict): Book data containing title, author and content

    Returns:
        str: Book title
    """
    from utils import set_cache_path, extract_json
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')

    save_dir = os.path.join(args.output_dir, 'total_scenes')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{book["title"]}.json')

    if os.path.exists(save_path) and not args.regenerate:
        return book['title']

    standardized_path = f'{args.output_dir}/standardized_merge_clean/{book["title"]}.json'
    if not os.path.exists(standardized_path):
        logger.warning(f"Standardized data not found for {book['title']}, skipping.")
        return book['title']

    with open(standardized_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    scenes = results.get('scenes', [])

    if not scenes:
        logger.warning(f"No scenes found for {book['title']}, skipping.")
        return book['title']

    logger.info(f"Enhancing scenes for: {book['title']} ({len(scenes)} scenes)")

    # Enhance scenarios/motivations for each scene
    for scene in tqdm(scenes, desc=f"Enhancing scenes for {book['title']}"):
        if scene is None:
            continue

        # Prepare input for scene enhancement
        input_scene = {
            'scene_summary': scene.get('summary', ''),
            'scenario': scene.get('scenario', ''),
            'interactions': scene.get('interactions', [])
        }

        # Generate prompt for enhancing scenario
        prompt = f"""
Given a scene from {book['title']}, enhance the scene setup to create a comprehensive foundation for dramatic performance, i.e., to provide necessary background for actors to act out the scene:

1. Review the provided scene and contextual details thoroughly.
2. Expand the 'scenario' with rich situational context that actors need to convincingly perform the scene. Focus on essential background information, while excluding future details to be portrayed in the scene.

===Output Format===
Please provide the output in the following JSON format:
{{
    "scenario": "A detailed scene-setting description that provides actors with essential context and atmosphere (< 200 words). Include all necessary background information while excluding future information to be revealed in the scene."
}}

===Requirements===
1. Adhere strictly to the specified output JSON format. 
2. [IMPORTANT] Ensure all DOUBLE QUOTES within all STRINGS are properly ESCAPED, especially when extracting from the text.
3. Output in the same language as the input. 

===Input Scene and Background===
{json.dumps(input_scene, ensure_ascii=False, indent=2)}
"""

        def parse_response(response, **kwargs):
            try:
                assert 'scenario' in response
                return response
            except:
                return False

        response = get_response_json(
            [extract_json, parse_response],
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_retry=5
        )

        # Update scene with enhanced scenario
        if response:
            scene['scenario'] = response['scenario']

    # Save enhanced scenes (preserve original structure, only scenes updated)
    results['scenes'] = scenes
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Enhanced scenes saved to {save_path}")
    return book['title']


def extract_location(book):
    """
    Extract locations mentioned in each scene of a book.

Reads from total_scenes/, extracts all locations (name + description)
    from each scene using LLM, and saves the result to locations/extracted/.

    Args:
        book (dict): Book data containing title, author and content

    Returns:
        str: Book title
    """
    from utils import set_cache_path, extract_json
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')

    save_dir = os.path.join(args.output_dir, 'locations_extracted')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{book["title"]}.json')

    if os.path.exists(save_path) and not args.regenerate:
        return book['title']

    source_path = os.path.join(args.output_dir, 'total_scenes', f'{book["title"]}.json')
    if not os.path.exists(source_path):
        logger.warning(f"total_scenes data not found for {book['title']}, skipping.")
        return book['title']

    with open(source_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    scenes = results.get('scenes', [])
    if not scenes:
        logger.warning(f"No scenes found for {book['title']}, skipping.")
        return book['title']

    logger.info(f"Extracting locations for: {book['title']} ({len(scenes)} scenes)")

    for scene in tqdm(scenes, desc=f"Extracting locations for {book['title']}"):
        if scene is None:
            continue

        input_scene = {
            'scenario': scene.get('scenario', ''),
            'interactions': scene.get('interactions', [])
        }

        prompt = f"""Given a scene from "{book['title']}", identify the single primary location where this scene takes place.

===Output Format===
Please provide the output in the following JSON format:
{{
    "name": "Location name",
    "description": "A brief description of this location (~30 words)"
}}

===Requirements===
1. Identify the main location where this scene takes place (e.g., a specific room, building, or place).
2. The name should be concise and specific; the description should briefly capture its key characteristics.
3. If no location is identifiable, set both name and description to empty strings.
4. Adhere strictly to the specified output JSON format.
5. [IMPORTANT] Ensure all DOUBLE QUOTES within all STRINGS are properly ESCAPED.
6. Output in the same language as the input.

===Input Scene===
{json.dumps(input_scene, ensure_ascii=False, indent=2)}
"""

        def parse_response(response, **kwargs):
            try:
                assert 'name' in response and 'description' in response
                return response
            except:
                return False

        response = get_response_json(
            [extract_json, parse_response],
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_retry=5
        )

        if response:
            scene['location'] = response
        else:
            scene['location'] = {'name': '', 'description': ''}

    results['scenes'] = scenes
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Extracted locations saved to {save_path}")
    return book['title']


def standardize_location_names(book):
    """
    Standardize location names across all scenes of a book.

    This function:
    1. Collects all locations from every scene (name + description, deduplicated by name)
    2. Sends them to an LLM in batches to identify aliases and determine official names
    3. Builds a unified official location list with aliases
    4. Replaces all location name references in scenes with official names
    5. Saves the result to locations/standardized/, with location_list appended

    Args:
        book (dict): Book data containing title, author, and content

    Returns:
        str: Book title
    """
    from utils import set_cache_path, extract_json
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')

    save_dir = os.path.join('dataset', 'extracted_data', 'scenes')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{book["title"]}.json')

    if os.path.exists(save_path) and not args.regenerate:
        return book['title']

    source_path = os.path.join(args.output_dir, 'locations_extracted', f'{book["title"]}.json')
    if not os.path.exists(source_path):
        logger.warning(f"Extracted locations data not found for {book['title']}, skipping.")        
        return book['title']

    with open(source_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    scenes = results.get('scenes', [])

    # ------------------------------------------------------------------ #
    # Step 1: Collect all unique locations (by name, first-occurrence     #
    #         description wins)                                            #
    # ------------------------------------------------------------------ #
    seen_names = {}  # name -> description
    for scene in scenes:
        loc = scene.get('location', {})
        if not loc:
            continue
        name = loc.get('name', '').strip()
        if not name:
            continue
        if name not in seen_names:
            seen_names[name] = loc.get('description', '')

    all_locations = [{'name': n, 'description': d} for n, d in seen_names.items()]
    all_input_names = set(seen_names.keys())

    logger.info(f"Total unique location names collected: {len(all_locations)}")

    # ------------------------------------------------------------------ #
    # Step 2: Send to LLM in batches, accumulating official list          #
    # ------------------------------------------------------------------ #
    BATCH_SIZE = 10

    official_location_list = []
    failed = False
    fail_prompts = []

    for batch_start in tqdm(range(0, len(all_locations), BATCH_SIZE), desc=f"Standardize locations {book['title']}"):
        batch = all_locations[batch_start: batch_start + BATCH_SIZE]
        batch_names = {loc['name'] for loc in batch}

        if official_location_list:
            existing_section = json.dumps(official_location_list, ensure_ascii=False, indent=2)
        else:
            existing_section = "[]"

        prompt = f"""You are helping to standardize location names in the book "{book['title']}".

## TASK
You will be given:
1. A list of **existing official locations** (already processed in previous batches) — may be empty on the first batch.
2. A list of **new locations** to process in this batch.

Your job is to produce an updated **official location list** by processing every new location in the batch.

## PRIORITY RULES (in order)

### PRIORITY 1 — Cover EVERY name (ABSOLUTE REQUIREMENT, zero exceptions)
Every single name from the new batch AND every name/alias from the existing official list MUST appear in the output — either as a `name` or inside `alias`. **Never drop any name under any circumstances.**
- This includes names that differ only in capitalization — they are treated as DIFFERENT names and ALL must be preserved.
- This also includes names that differ only by the presence or absence of "the" (e.g., "Forest" vs "The Forest") — they are treated as DIFFERENT names and ALL must be preserved.
- Before finalizing your output, do a coverage check: go through every input name one by one and confirm it appears in your output.

### PRIORITY 2 — Choose the most formal/complete name (best effort)
- If a new location clearly refers to the same place as an existing official location, merge them (add the new name as an alias, or upgrade the official name if the new name is more complete/formal).
- Collect all alternative names / short forms as `alias`.

## OUTPUT FORMAT (JSON)
{{
    "location_list": [
        {{
            "name": "Official location name (most complete/formal if known)",
            "description": "Brief location description (~20 words)",
            "alias": ["alternative name 1", "alternative name 2", ...]
        }}
    ]
}}

## REQUIREMENTS
1. Output MUST be valid JSON following the format above.
2. Every entry MUST have `name`, `description`, and `alias` fields (`alias` can be an empty list).
3. No duplicate entries — each real-world location appears exactly once.
4. The `alias` list must NOT include the official `name` itself.
5. **Coverage check (enforced)**: every name from the existing official list AND every name from the new batch MUST appear as either a `name` or in an `alias` in the output. **If even one name is missing, the output is invalid.**
6. **Exact name matching (strictly enforced)**: treat names with different capitalizations OR with/without leading "the" as DISTINCT names. ALL variants MUST be listed — do NOT silently drop any variant just because a similar name already exists. For example, "Forest", "forest", "The Forest", and "the forest" are four different names and all must be preserved if they appear in the input.
7. **Self-check before output**: mentally iterate through every input name and verify it is present in your output. Only output after this check passes.

## HIERARCHICAL LOCATION MERGING
Some locations are sub-locations contained within a larger location (e.g., a bedroom, garden, or living room that belongs to a specific home). Apply the following rules:
- If a new location is clearly a sub-location of an existing location (or another new location), **merge them into one entry**.
- Use the most complete/encompassing name as the official `name` (e.g., "XXX Home" rather than just "XXX bedroom").
- Add the sub-location names as `alias` entries.
- Update the `description` to mention that it contains those sub-locations (e.g., "...includes a garden, living room, and upstairs bedrooms").
- **Coverage still applies**: every sub-location name must appear in the `alias` list of its parent entry.
- **Do NOT merge locations just because their names look similar.** Always use BOTH the name AND the description to determine if two locations are truly the same place. For example, a family may have two "homes" — if their descriptions indicate different physical locations, they must remain as separate entries.

## EXISTING OFFICIAL LOCATIONS (from previous batches)
{existing_section}

## NEW LOCATIONS TO PROCESS (this batch)
{json.dumps(batch, ensure_ascii=False, indent=2)}
"""

        existing_names_to_cover = set()
        for entry in official_location_list:
            existing_names_to_cover.add(entry['name'])
            existing_names_to_cover.update(entry.get('alias', []))
        required_names = existing_names_to_cover | batch_names

        def parse_response(response, required_names, **kwargs):
            try:
                if 'location_list' not in response:
                    logger.warning("Response missing 'location_list' key")
                    return False
                loc_list = response['location_list']
                if not isinstance(loc_list, list):
                    return False
                for entry in loc_list:
                    if 'name' not in entry or 'alias' not in entry or 'description' not in entry:
                        logger.warning(f"Entry missing required fields: {entry}")
                        return False
                covered = set()
                for entry in loc_list:
                    covered.add(entry['name'])
                    covered.update(entry.get('alias', []))
                missing = required_names - covered
                if missing:
                    lower_to_entry = {}
                    for entry in loc_list:
                        lower_to_entry[entry['name'].lower()] = entry
                        for alias in entry.get('alias', []):
                            lower_to_entry[alias.lower()] = entry
                    still_missing = set()
                    for raw_name in missing:
                        matched_entry = lower_to_entry.get(raw_name.lower())
                        if matched_entry is None:
                            # Try toggling "the" prefix (case-insensitive)
                            raw_lower = raw_name.lower()
                            if raw_lower.startswith("the "):
                                alt_lower = raw_lower[4:]  # strip "the "
                            else:
                                alt_lower = "the " + raw_lower  # add "the "
                            matched_entry = lower_to_entry.get(alt_lower)
                        if matched_entry is not None:
                            if raw_name not in matched_entry['alias'] and raw_name != matched_entry['name']:
                                matched_entry['alias'].append(raw_name)
                            covered.add(raw_name)
                        else:
                            still_missing.add(raw_name)
                    if still_missing:
                        logger.warning(f"Coverage check failed — missing names: {still_missing}")
                        return False
                return response
            except Exception as e:
                logger.error(f"Error in parse_response for standardize_location_names: {e}")
                return False

        response = get_response_json(
            [extract_json, parse_response],
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            required_names=required_names,
            max_retry=5
        )

        if (not response or 'location_list' not in response) and args.candidate_model != args.model:
            logger.info(
                f"Primary model failed coverage/validation for batch {batch_start // BATCH_SIZE + 1}; retrying with candidate model {args.candidate_model}"
            )
            response = get_response_json(
                [extract_json, parse_response],
                model=args.candidate_model,
                messages=[{"role": "user", "content": prompt}],
                required_names=required_names,
                max_retry=5
            )

        if response and 'location_list' in response:
            official_location_list = response['location_list']
            logger.info(f"Batch {batch_start // BATCH_SIZE + 1}: official location list now has {len(official_location_list)} entries")
        else:
            logger.error(f"Failed to standardize location names for batch starting at {batch_start}")
            fail_prompts.append(prompt)
            failed = True
            break

    # ------------------------------------------------------------------ #
    # Step 3: If any batch failed, record and return early                #
    # ------------------------------------------------------------------ #
    if failed:
        # results.setdefault('fail', [])
        # for p in fail_prompts:
        #     results['fail'].append({'standardize_location_names_prompt': p})
        # with open(save_path, 'w', encoding='utf-8') as f:
        #     json.dump(results, f, ensure_ascii=False, indent=2)
        logger.warning(f"Location standardization failed for book: {book['title']}, partial results saved.")
        return book['title']

    # ------------------------------------------------------------------ #
    # Step 4: Build lookup table: any name/alias -> official name         #
    # ------------------------------------------------------------------ #
    name_to_official = {}
    for entry in official_location_list:
        official = entry['name']
        name_to_official[official] = official
        for alias in entry.get('alias', []):
            name_to_official[alias] = official

    uncovered = all_input_names - set(name_to_official.keys())
    if uncovered:
        logger.warning(f"The following location names were not covered by the official list: {uncovered}")

    # ------------------------------------------------------------------ #
    # Step 5: Replace names in all scenes                                 #
    # ------------------------------------------------------------------ #
    for scene in tqdm(scenes, desc=f"Replace location names {book['title']}"):
        loc = scene.get('location', {})
        if loc and loc.get('name', '').strip():
            raw_name = loc['name'].strip()
            scene['location']['name'] = name_to_official.get(raw_name, raw_name)

    # ------------------------------------------------------------------ #
    # Step 6: Save results with location_list appended                    #
    # ------------------------------------------------------------------ #
    results['scenes'] = scenes
    results = remove_private_scene_fields(results)

    ordered_results = {}
    for k, v in results.items():
        if k == 'fail':
            ordered_results['location_list'] = official_location_list
        ordered_results[k] = v
    if 'location_list' not in ordered_results:
        ordered_results['location_list'] = official_location_list

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(ordered_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Standardized location names saved to {save_path}")
    return book['title']


def world_initialization(book):
    """
    Build initial world cards for the book — both a global world card and
    location-dependent world cards — capturing the state of the world BEFORE
    any of the depicted events unfold.

    Global card: a single card describing the world at the start of the story,
    covering dimensions such as social norms, historical background, political
    landscape, cultural customs, economic systems, technology level, belief
    systems, important factions, key artifacts/objects, narrative style, etc.

    Location cards: one card per official location, describing that location's
    state at the start of the story, with fixed dimensions:
      - Detailed description of the location
      - Characters present at this location
      - Important entities (objects, institutions, etc.) and their states (NO characters — character cards are maintained separately)

    Args:
        book (dict): Book data containing title, author, and content

    Returns:
        str: Book title
    """
    
    from utils import set_cache_path, extract_json
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')

    save_dir = f'{args.output_dir}/world_initialization'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{book["title"]}.json'

    if os.path.exists(save_path) and not args.regenerate:
        return book['title']

    logger.info(f"Building world initialization cards for book: {book['title']}")

    # Load location-standardized scene data
    standardized_path = os.path.join('dataset', 'extracted_data', 'scenes', f'{book["title"]}.json')
    if not os.path.exists(standardized_path):
        logger.warning(f"Locations standardized data not found for {book['title']}, skipping.")
        return book['title']

    with open(standardized_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    scenes = results.get('scenes', [])
    location_list = results.get('location_list', [])

    if not scenes:
        logger.warning(f"No scenes found for {book['title']}, skipping.")
        return book['title']

    # Detect language from first scene text
    if len(scenes) > 0:
        from utils import lang_detect
        language = lang_detect(scenes[0].get('text', '')[:100])
        language = {'zh': 'Chinese', 'en': 'English'}.get(language, 'English')
    else:
        language = 'English'

    MAX_SCENE_TOKENS = 100000

    # ------------------------------------------------------------------ #
    # Part 1: Global World Card                                            #
    # ------------------------------------------------------------------ #

    global_prompt_template = """\
Build a **Global World Card**: the shared world-level backdrop for a story simulation system.

Character profiles and location cards already exist. This card covers **only** the global layer — stable world knowledge, systemic constraints, social logic, and broad conditions that shape behavior across the entire story.

**Goal**: Enable a model to role-play characters consistently and simulate plausible story evolution.

## Rules
1. Describe the world's **initial state BEFORE the story begins**. NO spoilers — no plot events, twists, deaths, or resolutions.
2. Infer backwards from the scenes: use them to deduce the world's norms, structures, and tensions, but do NOT narrate the scenes.
3. Focus on world-level context only. Skip details that belong in a character profile or location card unless they reflect a broader pattern.
4. Select only dimensions that matter for this book. Possible dimensions (use, skip, merge, or rename freely):
   - Social Order & Class · Historical Background & Tensions · Political Power & Institutions
   - Cultural Values & Moral Expectations · Family, Kinship, Duty & Reputation
   - Economy & Material Survival · Technology & Infrastructure
   - Religion, Belief & Ideology · Law & Social Consequences
   - Geography & Environmental Conditions · Important Factions & Organizations
   - Conflict Patterns & Behavioral Constraints · Special World Rules / Magic
   - Narrative Tone (only if critical for simulation)
5. Be concise. Every sentence must carry specific, useful information. No filler, no literary praise, no meta-commentary.
6. Stay grounded in the scene data and/or your knowledge of the book. Do not fabricate.
7. Output in {language}.

## Book
Title: {book_title}
Author: {book_author}

## Scene Data (reference only — do NOT narrate)
{scene_data}

Start directly with:
===GlobalWorldCard===
"""

    # Build scene data for global card (as many scenes as token budget allows)
    global_scene_list = []
    total_tokens = 0

    for scene in scenes:
        if scene is None:
            continue
        scene_entry = {
            'scene_summary': scene.get('summary', ''),
            'scenario': scene.get('scenario', ''),
            'interactions': scene.get('interactions', []),
        }
        entry_tokens = len(encode(json.dumps(scene_entry, ensure_ascii=False)))
        if total_tokens + entry_tokens > MAX_SCENE_TOKENS:
            if global_scene_list:
                logger.debug(
                    f"Global world card token budget reached, "
                    f"using {len(global_scene_list)}/{len(scenes)} scenes."
                )
                break
            else:
                # First scene exceeds budget — truncate interactions
                budget_for_interactions = MAX_SCENE_TOKENS - len(
                    encode(json.dumps({
                        'scene_summary': scene_entry['scene_summary'],
                        'scenario': scene_entry['scenario'],
                        'interactions': [],
                    }, ensure_ascii=False))
                )
                truncated_interactions = []
                inter_tokens = 0
                for inter in scene_entry['interactions']:
                    t = len(encode(json.dumps(inter, ensure_ascii=False)))
                    if inter_tokens + t > budget_for_interactions:
                        break
                    truncated_interactions.append(inter)
                    inter_tokens += t
                scene_entry['interactions'] = truncated_interactions
                global_scene_list.append(scene_entry)
                total_tokens += len(encode(json.dumps(scene_entry, ensure_ascii=False)))
                break
        global_scene_list.append(scene_entry)
        total_tokens += entry_tokens

    global_prompt = global_prompt_template.format(
        book_title=book['title'],
        book_author=book.get('author', 'Unknown'),
        language=language,
        scene_data=json.dumps(global_scene_list, ensure_ascii=False, indent=2),
    )

    logger.debug(f"Global world card prompt ({total_tokens} scene tokens):\n{global_prompt[:300]}...")

    nth_generation = 0
    global_card = ''
    while True:
        kwargs = dict(
            model=args.model,
            messages=[{"role": "user", "content": global_prompt}],
        )
        if nth_generation > 0:
            kwargs['nth_generation'] = nth_generation

        raw = get_response(**kwargs)

        try:
            if "===GlobalWorldCard===" in raw:
                global_card = raw.split("===GlobalWorldCard===", 1)[1].strip()
            else:
                global_card = raw.strip()
            if global_card.startswith('I apologize'):
                nth_generation += 1
                if nth_generation > 5:
                    logger.warning(f"Failed to generate global world card after 5 retries. Last response: {raw}")
                    global_card = ''
                    break
                continue
            break
        except Exception:
            nth_generation += 1
            if nth_generation > 5:
                logger.warning(f"Failed to generate global world card after 5 retries. Last response: {raw}")
                global_card = ''
                break
            continue

    logger.info(f"Global world card generated for {book['title']}")

    # ------------------------------------------------------------------ #
    # Part 2: Location-Dependent World Cards                               #
    # ------------------------------------------------------------------ #

    location_prompt_template = """\
Build a **Location World Card** for one specific location in a story simulation system.

A global world card and character cards already exist. This card covers **only** location-specific knowledge that complements them — what this place is like, how it works, and what matters inside it.

**Goal**: Enable a model to role-play characters at this location and simulate plausible interactions here.

## Rules
1. Describe this location's **initial state BEFORE the story begins**. NO spoilers.
2. Infer backwards from the scenes — deduce the place's character, layout, atmosphere, and important entities, but do NOT narrate the scenes.
3. **Important Entities must NOT include characters/people.** Character profiles are maintained in a separate system. Only include non-human entities such as objects, artifacts, animals, institutions, mechanisms, environmental features, etc.
4. Use all provided inputs:
   - `Name`: the official location.
   - `Description`: high-level information about it.
   - `Aliases`: alternative names merged into this location. Some may hint at **sub-locations** (e.g. bedrooms, gardens, offices within a home or estate). Treat these as useful clues, not an exhaustive list.
5. Stay grounded. Do not fabricate unsupported details.
6. Output in {language}. Keep JSON keys exactly as specified; write values in {language}.

## JSON Output — choose ONE structure

**Structure A (flat)** — use when sub-location grouping adds little value:
```json
{{
  "Detailed Description": "Concise, vivid description of the location's initial state: appearance, atmosphere, layout, sensory details, stable conditions.",
  "Important Entities": [
    {{"name": "entity name", "state": "initial condition / position"}}
  ]
}}
```

**Structure B (grouped)** — use when the location clearly contains important sub-locations:
```json
{{
  "Detailed Description": "Overall description of the location and how its sub-locations relate to the whole.",
  "Sub Locations": [
    {{
      "name": "sub-location name",
      "description": "what it is like and how it functions",
      "Important Entities": [
        {{"name": "entity name", "state": "initial condition / position"}}
      ]
    }}
  ]
}}
```

The presence of `Sub Locations` key distinguishes the two structures. Do NOT force sub-locations if weakly supported. Output valid JSON only — no text before or after.

## Book
Title: {book_title}

## Location
Name: {location_name}
Description: {location_description}
Aliases: {location_aliases}

## Scene Data (reference only — do NOT narrate)
{scene_data}

Start directly with:
===LocationWorldCard===
"""

    official_location_names = [entry['name'] for entry in location_list]

    # Build a map: official location name -> list of scene indices
    location_scene_map = {name: [] for name in official_location_names}

    for i_s, scene in enumerate(scenes):
        if scene is None:
            continue
        loc_data = scene.get('location')
        if not isinstance(loc_data, dict):
            continue
        scene_loc_name = loc_data.get('name', '')
        for name in official_location_names:
            if name == scene_loc_name:
                location_scene_map[name].append(i_s)

    # Build location description lookup
    location_desc_map = {entry['name']: entry.get('description', '') for entry in location_list}
    location_alias_map = {entry['name']: entry.get('alias', []) for entry in location_list}

    location_cards = {}

    for location_name in tqdm(official_location_names, desc=f"Building location cards {book['title']}", leave=False):
        involved_scene_indices = location_scene_map[location_name]

        if not involved_scene_indices:
            logger.info(f"No scenes found for location: {location_name}, skipping.")
            location_cards[location_name] = ''
            continue

        # Build scene data list, respecting token budget
        scene_data_list = []
        total_tokens = 0

        for i_s in involved_scene_indices:
            scene = scenes[i_s]
            if scene is None:
                continue

            scene_entry = {
                'scene_summary': scene.get('summary', ''),
                'scenario': scene.get('scenario', ''),
                'interactions': scene.get('interactions', []),
            }

            entry_tokens = len(encode(json.dumps(scene_entry, ensure_ascii=False)))

            if total_tokens + entry_tokens > MAX_SCENE_TOKENS:
                if scene_data_list:
                    logger.debug(
                        f"Token budget reached for location {location_name} at scene {i_s}, "
                        f"using {len(scene_data_list)}/{len(involved_scene_indices)} scenes."
                    )
                    break
                else:
                    budget_for_interactions = MAX_SCENE_TOKENS - len(
                        encode(json.dumps({
                            'scene_summary': scene_entry['scene_summary'],
                            'scenario': scene_entry['scenario'],
                            'interactions': [],
                        }, ensure_ascii=False))
                    )
                    truncated_interactions = []
                    inter_tokens = 0
                    for inter in scene_entry['interactions']:
                        t = len(encode(json.dumps(inter, ensure_ascii=False)))
                        if inter_tokens + t > budget_for_interactions:
                            break
                        truncated_interactions.append(inter)
                        inter_tokens += t
                    scene_entry['interactions'] = truncated_interactions
                    scene_data_list.append(scene_entry)
                    total_tokens += len(encode(json.dumps(scene_entry, ensure_ascii=False)))
                    break

            scene_data_list.append(scene_entry)
            total_tokens += entry_tokens

        location_prompt = location_prompt_template.format(
            book_title=book['title'],
            location_name=location_name,
            location_description=location_desc_map.get(location_name, ''),
            location_aliases=json.dumps(location_alias_map.get(location_name, []), ensure_ascii=False),
            language=language,
            scene_data=json.dumps(scene_data_list, ensure_ascii=False, indent=2),
        )

        logger.debug(f"Location card prompt for {location_name} ({total_tokens} scene tokens):\n{location_prompt[:300]}...")

        def _validate_location_card(card_dict):
            """Validate that a parsed location card dict has the required fields."""
            if not isinstance(card_dict, dict):
                return False, "Parsed result is not a dict"
            if "Detailed Description" not in card_dict:
                return False, "Missing required field 'Detailed Description'"
            if "Sub Locations" in card_dict:
                # Grouped structure: validate Sub Locations
                subs = card_dict["Sub Locations"]
                if not isinstance(subs, list) or len(subs) == 0:
                    return False, "'Sub Locations' must be a non-empty list"
                for i, sub in enumerate(subs):
                    if not isinstance(sub, dict):
                        return False, f"Sub Locations[{i}] is not a dict"
                    if "name" not in sub:
                        return False, f"Sub Locations[{i}] missing 'name'"
                    if "description" not in sub:
                        return False, f"Sub Locations[{i}] missing 'description'"
                    if "Important Entities" in sub:
                        ents = sub["Important Entities"]
                        if not isinstance(ents, list):
                            return False, f"Sub Locations[{i}]['Important Entities'] is not a list"
                        for j, ent in enumerate(ents):
                            if not isinstance(ent, dict) or "name" not in ent or "state" not in ent:
                                return False, f"Sub Locations[{i}].Important Entities[{j}] missing 'name' or 'state'"
            else:
                # Flat structure: validate Important Entities
                if "Important Entities" not in card_dict:
                    return False, "Missing 'Important Entities' (flat structure requires it when 'Sub Locations' is absent)"
                ents = card_dict["Important Entities"]
                if not isinstance(ents, list):
                    return False, "'Important Entities' is not a list"
                for j, ent in enumerate(ents):
                    if not isinstance(ent, dict) or "name" not in ent or "state" not in ent:
                        return False, f"Important Entities[{j}] missing 'name' or 'state'"
            return True, "OK"

        nth_generation = 0
        location_card = None
        while True:
            kwargs = dict(
                model=args.model,
                messages=[{"role": "user", "content": location_prompt}],
            )
            if nth_generation > 0:
                kwargs['nth_generation'] = nth_generation

            raw = get_response(**kwargs)

            try:
                if "===LocationWorldCard===" in raw:
                    card_text = raw.split("===LocationWorldCard===", 1)[1].strip()
                else:
                    card_text = raw.strip()

                if card_text.startswith('I apologize'):
                    nth_generation += 1
                    if nth_generation > 5:
                        logger.warning(f"Failed to generate location card for {location_name} after 5 retries. Last response: {raw}")
                        location_card = None
                        break
                    continue

                # Parse JSON
                parsed = extract_json(card_text, model=args.model)
                if parsed is None:
                    logger.warning(f"JSON extraction failed for {location_name} (attempt {nth_generation + 1}). Raw: {card_text[:300]}...")
                    nth_generation += 1
                    if nth_generation > 5:
                        logger.warning(f"Failed to parse location card JSON for {location_name} after 5 retries.")
                        location_card = None
                        break
                    continue

                # Validate fields
                valid, reason = _validate_location_card(parsed)
                if not valid:
                    logger.warning(f"Location card validation failed for {location_name} (attempt {nth_generation + 1}): {reason}")
                    nth_generation += 1
                    if nth_generation > 5:
                        logger.warning(f"Failed to generate valid location card for {location_name} after 5 retries. Last reason: {reason}")
                        location_card = None
                        break
                    continue

                location_card = parsed
                break

            except Exception as e:
                logger.warning(f"Exception processing location card for {location_name} (attempt {nth_generation + 1}): {e}")
                nth_generation += 1
                if nth_generation > 5:
                    logger.warning(f"Failed to generate location card for {location_name} after 5 retries. Last exception: {e}")
                    location_card = None
                    break
                continue

        location_cards[location_name] = location_card

    # ------------------------------------------------------------------ #
    # Save results                                                         #
    # ------------------------------------------------------------------ #
    output = {
        'global_world_card': global_card,
        'location_cards': location_cards,
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"World initialization cards saved to {save_path}")
    return book['title']


def world_dynamic(book):
    """
    Dynamically update world cards (global + per-location) as the story progresses,
    interaction by interaction within each scene.

    For each scene (each scene has exactly one location), this function:
    1. Batches the interactions (with a configurable batch size).
    2. Numbers each interaction as S{scene_index}-I{interaction_index}.
    3. Provides a lookahead of future interactions so the model can judge consequences.
    4. Asks the model to decide, for each interaction in the batch, whether to update
       the global world card and/or the current location's world card.
    5. Only outputs updated cards for interactions that need updates.
    6. At the start and after each scene ends, generates a short location description
       for later simulation-time location selection.

    Output is saved to dataset/extracted_data/world_dynamic/{book_title}.json with structure:
    {
        "global_card_history": [
            {
                "scene_index": int,
                "interaction_index": int,
                "global_card": str
            }
        ],
        "location_cards": {
            "location_name": {
                "card_history": [
                    {
                        "scene_index": int,
                        "interaction_index": int,
                        "location_card": dict   // same structure as initialization
                    }
                ],
                "scene_descriptions": [
                    {
                        "scene_index": int,      // -1 for the initial description
                        "description": str       // short description for simulation
                    }
                ]
            }
        }
    }

    Args:
        book (dict): Book data containing title, author, and content

    Returns:
        str: Book title
    """
    from utils import set_cache_path, extract_json
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')

    save_dir = os.path.join('dataset', 'extracted_data', 'world_dynamic')
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{book["title"]}.json'

    if os.path.exists(save_path) and not args.regenerate:
        return book['title']

    logger.info(f"Building dynamic world cards for book: {book['title']}")

    # ------------------------------------------------------------------ #
    # Step 1: Load data                                                    #
    # ------------------------------------------------------------------ #
    standardized_path = os.path.join('dataset', 'extracted_data', 'scenes', f'{book["title"]}.json')
    if not os.path.exists(standardized_path):
        logger.warning(f"Locations standardized data not found for {book['title']}, skipping.")
        return book['title']

    init_world_path = f'{args.output_dir}/world_initialization/{book["title"]}.json'
    if not os.path.exists(init_world_path):
        logger.warning(f"World initialization data not found for {book['title']}, skipping.")
        return book['title']

    with open(standardized_path, 'r', encoding='utf-8') as f:
        standardized_data = json.load(f)

    with open(init_world_path, 'r', encoding='utf-8') as f:
        init_world = json.load(f)

    scenes = standardized_data.get('scenes', [])
    location_list = standardized_data.get('location_list', [])

    if not scenes:
        logger.warning(f"No scenes found for {book['title']}, skipping.")
        return book['title']

    # Detect language
    if len(scenes) > 0:
        from utils import lang_detect
        language = lang_detect(scenes[0].get('text', '')[:100])
        language = {'zh': 'Chinese', 'en': 'English'}.get(language, 'English')
    else:
        language = 'English'

    # Load initial cards
    current_global_card = init_world.get('global_world_card', '')
    init_location_cards = init_world.get('location_cards', {})

    official_location_names = [entry['name'] for entry in location_list]
    location_desc_map = {entry['name']: entry.get('description', '') for entry in location_list}

    # Current state for each location card (mutable)
    current_location_cards = {}
    for loc_name in official_location_names:
        card = init_location_cards.get(loc_name)
        current_location_cards[loc_name] = card if card else {}

    INTERACTION_BATCH_SIZE = 30
    LOOKAHEAD_INTERACTIONS = 10

    # ------------------------------------------------------------------ #
    # Step 2: Prompt templates                                             #
    # ------------------------------------------------------------------ #

    # Prompt for generating initial location description (before any scene)
    init_location_desc_prompt_template = """\
Write a short description (30–60 words) of the location "{location_name}" for a story simulation system. This description will be shown alongside all other location descriptions to help decide where the next scene takes place.

## Context
**Book:** {book_title}
**Location card:**
{location_card}
**Location's basic info:** {location_basic_desc}

## Requirements
1. Cover: what this place IS, its atmosphere/function, and what kinds of events or interactions typically happen here.
2. Write in third person, present tense. Be concrete and specific.
3. Output ONLY the description text — no JSON, no labels, no preamble.
4. Output in {language}.
"""

    # Prompt for generating post-scene location description
    post_scene_location_desc_prompt_template = """\
Write a short description (30–60 words) of the location "{location_name}" AS OF THE END OF Scene #{scene_index}, for a story simulation system. This description will be shown alongside all other location descriptions to help decide where the next scene takes place.

## Context
**Book:** {book_title}
**Current location card:**
{location_card}
**Scene #{scene_index} summary:** {scene_summary}

## Requirements
1. Cover: what this place IS now, its current atmosphere, and what just happened here that affects its state.
2. Write in third person, present tense. Be concrete and specific.
3. Do NOT reveal future plot events. Reflect only the state as of the end of this scene.
4. Output ONLY the description text — no JSON, no labels, no preamble.
5. Output in {language}.
"""

    # Main prompt: per-batch world card update
    world_update_prompt_template = """\
You are tracking how the world evolves in a story, interaction by interaction. Your goal is to maintain an accurate, up-to-date global world card and location world card as events unfold — providing the necessary world context for actors to role-play characters and for the simulation to evolve the story plausibly.

## CURRENT WORLD STATE

**Global World Card** (the world's current systemic state):
{current_global_card}

**Current Location:** {location_name}
**Current Location Card:**
{current_location_card}

## SCENE #{scene_index}
**Summary:** {scene_summary}
**Scenario:** {scene_scenario}

## INTERACTIONS IN THIS BATCH
Each interaction is numbered with a simple index (0, 1, 2, …). Evaluate EACH interaction for whether it impacts the world state.

{batch_interactions}

## LOOKAHEAD (reference ONLY — use to judge whether the current interaction has impacted the world; do NOT reveal or incorporate these future events into the cards)

### Global Card Lookahead (next sequential interactions, possibly from the next scene)
{lookahead_global}

### Location Card Lookahead (next interactions at this same location, possibly from a future scene)
{lookahead_location}

**IMPORTANT**: The LOOKAHEAD section is strictly for reference. Do NOT include any information from the lookahead interactions in your card updates. Do NOT produce updates for any lookahead interaction — only update cards for interactions in the current batch.

## YOUR TASKS

For each interaction in this batch, decide:
1. **Should the global world card be updated?** Update only when an interaction causes a meaningful, lasting change to the world's systemic state (e.g., a power shift, a social norm broken, a faction destroyed, a new law enacted, an economic upheaval). Do NOT update for character-level events that don't affect the broader world.
2. **Should the current location card be updated?** Update only when an interaction causes a meaningful, lasting change to the physical environment, atmosphere, or important entities at this location (e.g., an object destroyed, a room's state altered, a new entity introduced, a major environmental change). Do NOT update for transient actions that leave no lasting mark.

**Reminder:** "Important Entities" in location cards must NEVER include characters/people. Character profiles are maintained separately. Only track non-human entities (objects, artifacts, animals, institutions, mechanisms, environmental features, etc.).

**Card Maintenance Principle:** Updating a card is NOT simply appending new information. You must also:
- **Remove outdated or superseded information** — if a previous state is no longer true (e.g., a building was destroyed, a political regime was overthrown, an object was taken away), delete or replace the old description rather than keeping both old and new.
- **Use concise, summarized language** — describe world states in brief, high-level terms. Avoid verbose narratives or blow-by-blow recounting of events. The card should capture the *current state* of the world/location, not a history log.
- **Keep cards compact** — the card should NOT grow indefinitely as the story progresses. Consolidate and compress information when updating.

## OUTPUT FORMAT (JSON)
Only include interactions that trigger at least one update. If no interaction in this batch triggers any update, output empty lists.
{{
    "global": [
        {{
            "interaction_id": <integer index of the interaction in this batch>,
            "card": "Full updated global world card as a Markdown string"
        }}
    ],
    "location": [
        {{
            "interaction_id": <integer index of the interaction in this batch>,
            "card": <Full updated location card as a JSON object (same structure as input)>
        }}
    ]
}}

## RULES
1. Output MUST be valid JSON with exactly two keys: "global" and "location", each being a list.
2. Only include entries for interactions that actually trigger an update. Omit interactions that cause no change.
3. When updating a card, output the COMPLETE updated card (not a diff). Reflect the current world state accurately — this means modifying what changed, removing what is no longer true, and keeping what still holds. Do NOT blindly preserve all old information.
4. Be highly selective — most interactions should NOT trigger updates. Only update when there is a genuine, lasting change to the world or location state.
5. Keep cards concise and compact. Use summarized, high-level descriptions rather than detailed event narrations. The card should read like a current-state snapshot, not a chronological log. If the card is growing too long, consolidate and compress older entries.
5. The location card JSON must follow the same structure as the input (flat with "Detailed Description" + "Important Entities", or grouped with "Detailed Description" + "Sub Locations").
6. `card` in "global" must be a plain Markdown string. `card` in "location" must be a JSON object.
7. Do NOT produce updates for any interaction outside this batch (especially not for lookahead interactions).
8. Output in {language}.
"""

    # ------------------------------------------------------------------ #
    # Step 3: Initialize output structures                                 #
    # ------------------------------------------------------------------ #
    global_card_history = [{
        'scene_index': 0,
        'interaction_index': -1,
        'global_card': current_global_card,
    }]

    location_card_data = {}
    for loc_name in official_location_names:
        location_card_data[loc_name] = {
            'card_history': [{
                'scene_index': 0,
                'interaction_index': -1,
                'location_card': current_location_cards.get(loc_name, {}),
            }],
            'scene_descriptions': [],
        }

    # ------------------------------------------------------------------ #
    # Step 4: Generate initial location descriptions                       #
    # ------------------------------------------------------------------ #
    for loc_name in tqdm(official_location_names, desc=f"Init location descs {book['title']}", leave=False):
        loc_card = current_location_cards.get(loc_name, {})
        loc_card_str = json.dumps(loc_card, ensure_ascii=False, indent=2) if isinstance(loc_card, dict) else str(loc_card)

        init_desc_prompt = init_location_desc_prompt_template.format(
            location_name=loc_name,
            book_title=book['title'],
            location_card=loc_card_str,
            location_basic_desc=location_desc_map.get(loc_name, ''),
            language=language,
        )

        nth_generation = 0
        init_desc = ''
        while True:
            kwargs = dict(
                model=args.model,
                messages=[{"role": "user", "content": init_desc_prompt}],
            )
            if nth_generation > 0:
                kwargs['nth_generation'] = nth_generation

            raw = get_response(**kwargs)
            raw = raw.strip()
            if raw and not raw.startswith('I apologize'):
                init_desc = raw
                break
            nth_generation += 1
            if nth_generation > 5:
                logger.warning(f"Failed to generate initial description for {loc_name}")
                break

        location_card_data[loc_name]['scene_descriptions'].append({
            'scene_index': -1,
            'description': init_desc,
        })

    # ------------------------------------------------------------------ #
    # Step 5: Process scene by scene, batch by batch                       #
    # ------------------------------------------------------------------ #
    for i_s, scene in enumerate(tqdm(scenes, desc=f"World dynamic {book['title']}", leave=False)):
        if scene is None:
            continue

        interactions = scene.get('interactions', [])
        if not interactions:
            continue

        # Determine this scene's location
        loc_data = scene.get('location')
        if not isinstance(loc_data, dict):
            continue
        scene_loc_name = loc_data.get('name', '')
        has_valid_location = scene_loc_name in current_location_cards
        if not has_valid_location:
            logger.warning(f"Scene {i_s} location '{scene_loc_name}' not in official list, will only update global card.")
            current_loc_card = None
        else:
            current_loc_card = current_location_cards[scene_loc_name]

        # Number interactions
        numbered_interactions = []
        for i_inter, inter in enumerate(interactions):
            numbered_interactions.append({
                'id': f'S{i_s}-I{i_inter}',
                'characters': inter.get('characters', []),
                'content': inter.get('content', ''),
            })

        # Process in batches
        num_batches = (len(numbered_interactions) + INTERACTION_BATCH_SIZE - 1) // INTERACTION_BATCH_SIZE

        for batch_idx in range(num_batches):
            batch_start = batch_idx * INTERACTION_BATCH_SIZE
            batch_end = min(batch_start + INTERACTION_BATCH_SIZE, len(numbered_interactions))
            batch = numbered_interactions[batch_start:batch_end]

            # Build batch interactions string (use simple 0-based index in prompt)
            batch_lines = []
            for local_idx, item in enumerate(batch):
                chars_str = ', '.join(item['characters']) if item['characters'] else '(narrator)'
                batch_lines.append(f"**[{local_idx}]** [{chars_str}]: {item['content']}")
            batch_interactions_str = '\n\n'.join(batch_lines)

            # Build lookahead: next LOOKAHEAD_INTERACTIONS interactions after this batch
            # We build two lookaheads:
            #   1. lookahead_global: next sequential interactions (for global card updates)
            #   2. lookahead_location: next interactions at the SAME location (for location card updates)
            lookahead_start = batch_end
            lookahead_end = min(lookahead_start + LOOKAHEAD_INTERACTIONS, len(numbered_interactions))

            is_last_batch_in_scene = (batch_end >= len(numbered_interactions))

            if not is_last_batch_in_scene:
                # Not the last batch: both lookaheads are the same (remaining interactions in this scene)
                lookahead_lines = []
                for item in numbered_interactions[lookahead_start:lookahead_end]:
                    chars_str = ', '.join(item['characters']) if item['characters'] else '(narrator)'
                    lookahead_lines.append(f"**[{item['id']}]** [{chars_str}]: {item['content']}")
                shared_str = '\n\n'.join(lookahead_lines) if lookahead_lines else '(No further interactions available)'
                lookahead_global_str = shared_str
                lookahead_location_str = shared_str
            else:
                # Last batch in scene: build separate lookaheads

                # --- Global lookahead: next sequential scene --- #
                global_lookahead_lines = []
                global_lookahead_scene_header = ''
                for future_s in range(i_s + 1, len(scenes)):
                    if scenes[future_s] is None:
                        continue
                    future_scene = scenes[future_s]
                    future_summary = future_scene.get('summary', '')
                    future_scenario = future_scene.get('scenario', '')
                    global_lookahead_scene_header = f'(From next scene S{future_s})'
                    if future_summary:
                        global_lookahead_scene_header += f'\n**Summary:** {future_summary}'
                    if future_scenario:
                        global_lookahead_scene_header += f'\n**Scenario:** {future_scenario}'
                    future_inters = future_scene.get('interactions', [])
                    for li, finter in enumerate(future_inters[:LOOKAHEAD_INTERACTIONS]):
                        f_chars = ', '.join(finter.get('characters', [])) if finter.get('characters') else '(narrator)'
                        global_lookahead_lines.append(f"**[S{future_s}-I{li}]** [{f_chars}]: {finter.get('content', '')}")
                    if global_lookahead_lines:
                        break  # only peek into the first available next scene
                if global_lookahead_lines:
                    lookahead_global_str = global_lookahead_scene_header + '\n' + '\n\n'.join(global_lookahead_lines)
                else:
                    lookahead_global_str = '(No further interactions available — end of story)'

                # --- Location lookahead: next scene at the SAME location --- #
                location_lookahead_lines = []
                location_lookahead_scene_header = ''
                for future_s in range(i_s + 1, len(scenes)):
                    if scenes[future_s] is None:
                        continue
                    future_loc = scenes[future_s].get('location')
                    if not isinstance(future_loc, dict):
                        continue
                    if has_valid_location and future_loc.get('name', '') == scene_loc_name:
                        future_scene = scenes[future_s]
                        future_summary = future_scene.get('summary', '')
                        future_scenario = future_scene.get('scenario', '')
                        location_lookahead_scene_header = f'(From next scene at this location: S{future_s})'
                        if future_summary:
                            location_lookahead_scene_header += f'\n**Summary:** {future_summary}'
                        if future_scenario:
                            location_lookahead_scene_header += f'\n**Scenario:** {future_scenario}'
                        future_inters = future_scene.get('interactions', [])
                        for li, finter in enumerate(future_inters[:LOOKAHEAD_INTERACTIONS]):
                            f_chars = ', '.join(finter.get('characters', [])) if finter.get('characters') else '(narrator)'
                            location_lookahead_lines.append(f"**[S{future_s}-I{li}]** [{f_chars}]: {finter.get('content', '')}")
                        break  # only peek into the next scene at this location
                if location_lookahead_lines:
                    lookahead_location_str = location_lookahead_scene_header + '\n' + '\n\n'.join(location_lookahead_lines)
                else:
                    lookahead_location_str = '(No further interactions at this location)'

            # Build current location card string
            if has_valid_location:
                loc_card_str = json.dumps(current_loc_card, ensure_ascii=False, indent=2) if isinstance(current_loc_card, dict) else str(current_loc_card)
            else:
                loc_card_str = '(No location card — this scene has no recognized location)'

            update_prompt = world_update_prompt_template.format(
                current_global_card=current_global_card if current_global_card else '(No global card yet)',
                location_name=scene_loc_name if has_valid_location else '(Unknown / Not Applicable)',
                current_location_card=loc_card_str,
                scene_index=i_s,
                scene_summary=scene.get('summary', ''),
                scene_scenario=scene.get('scenario', ''),
                batch_interactions=batch_interactions_str,
                lookahead_global=lookahead_global_str,
                lookahead_location=lookahead_location_str,
                language=language,
            )

            # Parse response
            valid_local_ids = set(range(len(batch)))

            # Prefixes the model might copy from the prompt template
            _GLOBAL_CARD_PREFIXES = [
                "## CURRENT WORLD STATE",
                "**Global World Card** (the world's current systemic state):",
                "**Global World Card**:",
                "Global World Card:",
            ]

            def _clean_global_card(card_text):
                """Strip prompt-template prefixes that the model sometimes copies into the global card output."""
                cleaned = card_text.strip()
                changed = True
                while changed:
                    changed = False
                    for prefix in _GLOBAL_CARD_PREFIXES:
                        if cleaned.startswith(prefix):
                            cleaned = cleaned[len(prefix):].strip()
                            changed = True
                return cleaned

            def parse_world_update_response(response, **kwargs):
                try:
                    if 'global' not in response or 'location' not in response:
                        logger.warning("World update response missing 'global' or 'location' key")
                        return False
                    if not isinstance(response['global'], list) or not isinstance(response['location'], list):
                        logger.warning("'global' or 'location' is not a list")
                        return False
                    for g in response['global']:
                        if 'interaction_id' not in g or 'card' not in g:
                            logger.warning("Global update entry missing 'interaction_id' or 'card'")
                            return False
                        iid = g['interaction_id']
                        if not isinstance(iid, int) or iid not in valid_local_ids:
                            logger.warning(f"Invalid global interaction_id: {iid}")
                            return False
                        if not g['card'] or not isinstance(g['card'], str):
                            logger.warning(f"Global card is empty or not a string for interaction {iid}")
                            return False
                        # Clean prompt-template prefixes from global card
                        g['card'] = _clean_global_card(g['card'])
                    if has_valid_location:
                        for l in response['location']:
                            if 'interaction_id' not in l or 'card' not in l:
                                logger.warning("Location update entry missing 'interaction_id' or 'card'")
                                return False
                            iid = l['interaction_id']
                            if not isinstance(iid, int) or iid not in valid_local_ids:
                                logger.warning(f"Invalid location interaction_id: {iid}")
                                return False
                            lc = l['card']
                            if not isinstance(lc, dict):
                                logger.warning(f"location card is not a dict for interaction {iid}")
                                return False
                            if 'Detailed Description' not in lc:
                                logger.warning(f"location card missing 'Detailed Description' for interaction {iid}")
                                return False
                    else:
                        # No valid location — discard any location updates the model may have produced
                        response['location'] = []
                    return response
                except Exception as e:
                    logger.error(f"Error parsing world update response: {e}")
                    return False

            update_response = get_response_json(
                [extract_json, parse_world_update_response],
                model=args.model,
                messages=[{"role": "user", "content": update_prompt}],
                max_retry=5,
            )

            if not update_response:
                logger.warning(f"Failed to get world update for scene {i_s} batch {batch_idx}, retrying with candidate model")
                update_response = get_response_json(
                    [extract_json, parse_world_update_response],
                    model=args.candidate_model,
                    messages=[{"role": "user", "content": update_prompt}],
                    max_retry=5,
                )

            if update_response:
                # Process global updates
                for g in update_response['global']:
                    local_idx = g['interaction_id']
                    # Convert local batch index to absolute interaction index
                    i_inter = batch_start + local_idx
                    current_global_card = g['card']
                    global_card_history.append({
                        'scene_index': i_s,
                        'interaction_index': i_inter,
                        'global_card': current_global_card,
                    })
                    logger.debug(f"Global card updated at S{i_s}-I{i_inter}")

                # Process location updates (only if this scene has a valid location)
                if has_valid_location:
                    for l in update_response['location']:
                        local_idx = l['interaction_id']
                        i_inter = batch_start + local_idx
                        current_loc_card = l['card']
                        current_location_cards[scene_loc_name] = current_loc_card
                        location_card_data[scene_loc_name]['card_history'].append({
                            'scene_index': i_s,
                            'interaction_index': i_inter,
                            'location_card': current_loc_card,
                        })
                        logger.debug(f"Location card '{scene_loc_name}' updated at S{i_s}-I{i_inter}")
            else:
                logger.error(f"Failed to get world update for scene {i_s} batch {batch_idx} with both models, aborting book: {book['title']}")
                return None

        # ---- After each scene: generate post-scene location description ---- #
        if not has_valid_location:
            logger.debug(f"Scene {i_s} has no valid location, skipping post-scene location description.")
            continue

        loc_card_str = json.dumps(current_loc_card, ensure_ascii=False, indent=2) if isinstance(current_loc_card, dict) else str(current_loc_card)

        post_desc_prompt = post_scene_location_desc_prompt_template.format(
            location_name=scene_loc_name,
            book_title=book['title'],
            scene_index=i_s,
            location_card=loc_card_str,
            scene_summary=scene.get('summary', ''),
            language=language,
        )

        nth_generation = 0
        post_desc = ''
        while True:
            kwargs = dict(
                model=args.model,
                messages=[{"role": "user", "content": post_desc_prompt}],
            )
            if nth_generation > 0:
                kwargs['nth_generation'] = nth_generation

            raw = get_response(**kwargs)
            raw = raw.strip()
            if raw and not raw.startswith('I apologize'):
                post_desc = raw
                break
            nth_generation += 1
            if nth_generation > 5:
                logger.warning(f"Failed to generate post-scene description for {scene_loc_name} after scene {i_s}")
                break

        location_card_data[scene_loc_name]['scene_descriptions'].append({
            'scene_index': i_s,
            'description': post_desc,
        })

    # ------------------------------------------------------------------ #
    # Step 6: Save results                                                 #
    # ------------------------------------------------------------------ #
    output = {
        'global_card_history': global_card_history,
        'location_cards': location_card_data,
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Dynamic world cards saved to {save_path}")
    return book['title']


def process_book(book):
    """Process a single book through extraction and cache restoration.
    
    Args:
        book (dict): Book data containing title, author, and content
        
    Returns:
        str or None: Book title if successful, None if failed
    """
    try:
        extract(book)
        result = restore_from_cache(book)
        result = clean_scenes(book)
        result = standardize_character_names(book)
        result = merge_interactions_for_book(book)
        result = clean_duplicate_scenes_for_book(book)
        result = build_profiles_initialization(book)
        result = build_profiles_dynamic(book)
        result = enhance_scenes(book)
        result = extract_location(book)
        result = standardize_location_names(book)
        result = world_initialization(book)
        result = world_dynamic(book)
        if result is None:
            logger.warning(f"Book processing aborted: {book.get('title', 'Unknown')}")
            return None
        logger.info(f"Successfully processed book: {book.get('title', 'Unknown')}")
        return result
    except Exception as e:
        logger.error(f"Error processing book {book.get('title', 'Unknown')}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


if __name__ == '__main__':

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read input data
    with jsonlines.open(args.input, mode='r') as reader:
        books_data = list(reader)

    # books_data = books_data[:3]
    # books_data = [books_data[24]]

    # Clean book titles
    for book in books_data:
        book['title'] = book['title'].replace('/', '-').replace(':', '_').replace('.', ' ')

    logger.info(f"Processing {len(books_data)} books")

    if args.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        
        logger.info(f"Starting parallel processing with {args.num_workers} workers")

        # Process books in parallel
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            processed_books = list(tqdm(
                executor.map(process_book, books_data),
                total=len(books_data),
                desc="Processing books"
            ))
    else:
        processed_books = []
        for book in tqdm(books_data):
            processed_book = process_book(book)
            processed_books.append(processed_book)














