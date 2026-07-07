
import re 
import argparse
from typing import List, Tuple
from collections import Counter
import json
import jsonlines
from utils import cached

count_split_success = 0
count_split_failed = 0

def find_content_start(content: str, book_title: str):
    """
    Iteratively find the start position of main content.
    
    Args:
    content (str): The full book content.
    book_title (str): The title of the book.
    
    Returns:
    int: The start position of main content if found in this content.
    None: If the main content start is not found in this content.
    """
    from utils import get_response, logger, extract_json
    import json
    
    chunk_size = 20000
    max_iterations = 5
    current_pos = 0
    
    for iteration in range(max_iterations):
        # Get a chunk from current position
        chunk_end = min(current_pos + chunk_size, len(content))
        chunk = content[current_pos:chunk_end]
        
        prompt = f"""You are analyzing the book "{book_title}".

Analyze this text segment:
{chunk}

Your task is to determine if this text segment contains the START of the MAIN CONTENT (the formal story/narrative plot).

You need to EXCLUDE and skip over:
- Prefaces, forewords, introductions by editors/publishers
- Copyright notices, publication information, ISBN, publisher info
- Table of contents
- Dedications (unless very brief and part of the narrative flow)
- Author's notes, translator's notes before the story
- Any non-narrative front matter

INCLUDE as the start (only formal narrative content):
- Prologues that are part of the formal story plot
- The first chapter or beginning of the formal narrative (MUST include the chapter number/title like "Chapter 1", "I", "Prologue", etc.)
- Any text that begins the actual formal story plot

**CRITICAL REQUIREMENTS for start_marker**:
1. The start_marker MUST be copied EXACTLY from the text, preserving ALL characters including:
   - Spaces (leading, trailing, and between words)
   - Newline characters (\n)
   - Tab characters
   - Any special characters
2. The marker MUST include the first chapter heading/number/title if present
3. The marker should be 10-20 words long to ensure uniqueness
4. Copy the text DIRECTLY - do not paraphrase or modify it in any way

Examples:
- If text has "Chapter 1 It was a dark", copy exactly: "Chapter 1 It was a dark"
- If text has "  I. Call me Ishmael", copy exactly: "  I. Call me Ishmael"
Do not add or skip any space.

Respond in JSON format:
{{
    "contains_start": true/false,
    "start_marker": "EXACT text snippet copied character-by-character from the text above, or null if not found",
    "reasoning": "brief explanation"
}}"""

        try:
            response = get_response(model='gemini-2.5-pro', messages=prompt, nth_generation=0)
            if not response:
                logger.warning(f"Failed to get LLM response for start detection")
                return current_pos
            
            result = extract_json(response)
            contains_start = result.get('contains_start', False)
            start_marker = result.get('start_marker')
            reasoning = result.get('reasoning', '')
            
            logger.info(f"Start detection iteration {iteration + 1}: {reasoning}")
            
            if contains_start and start_marker:
                # contains_start=true: the story starts in this content, do NOT advance to next chunk
                # Find the exact marker position with retry mechanism
                start_marker = start_marker.strip()
                marker_pos = content.find(start_marker, current_pos)
                
                # Retry up to 5 times if marker not found
                retry_count = 0
                max_retries = 5
                while marker_pos == -1 and retry_count < max_retries:
                    retry_count += 1
                    logger.warning(f"Start marker not found in content for '{book_title}' (attempt {retry_count}/{max_retries}): {start_marker[:50]}")
                    logger.info(f"Retrying to get a better start marker...")
                    
                    # Ask LLM again for a better marker
                    try:
                        response = get_response(model='gemini-2.5-pro', messages=prompt, nth_generation=retry_count)
                        if response:
                            result = extract_json(response)
                            new_marker = result.get('start_marker')
                            if new_marker:
                                start_marker = new_marker.strip()
                                marker_pos = content.find(start_marker, current_pos)
                    except Exception as e:
                        logger.error(f"Error during retry {retry_count}: {e}")
                        break
                
                if marker_pos != -1:
                    logger.info(f"Found start marker at position {marker_pos} (after {retry_count} retries)")
                    return marker_pos
                else:
                    logger.warning(f"Start marker still not found after {max_retries} retries, using current_pos {current_pos}")
                    return current_pos  # contains_start is true, stop here even if marker not found
            
            # contains_start is false: story does not start in this chunk, move to next chunk
            current_pos = chunk_end
            if current_pos >= len(content):
                break
        except Exception as e:
            logger.error(f"Error in find_content_start iteration {iteration}: {e}")
            break
    
    return None


def find_content_end(content: str, book_title: str, start_pos: int):
    """
    Iteratively find the end position of main content, searching backwards from the end.
    
    Args:
    content (str): The full book content.
    book_title (str): The title of the book.
    start_pos (int): The start position of main content.
    
    Returns:
    int: The end position of main content if found in this content.
    None: If the main content end is not found in this content.
    """
    from utils import get_response, logger, extract_json
    import json
    
    chunk_size = 20000
    max_iterations = 5
    current_pos = len(content)
    
    for iteration in range(max_iterations):
        # Get a chunk from current position, going backwards
        chunk_start = max(start_pos, current_pos - chunk_size)
        chunk = content[chunk_start:current_pos]
        
        prompt = f"""You are analyzing the book "{book_title}".

Analyze this text segment (this is from near the END of the book):
{chunk}

Your task is to determine if this text segment contains the END of the MAIN CONTENT (the formal story/narrative plot).

You need to EXCLUDE and identify where to cut off:
- Afterwords, postscripts by editors/publishers
- Epilogues written as diary entries, letters, or informal notes AFTER the main plot concludes
- Author's notes, translator's notes, editor's notes at the end
- Reader comments, reviews, or discussions
- Appendices (unless they are part of the formal narrative)
- "About the author" sections
- Advertisements for other books
- Publication information at the end
- Any informal supplementary content added after the story ends

INCLUDE as part of the main content (only formal narrative):
- Epilogues that are written as formal narrative prose and directly continue the story plot
- The final chapter or conclusion of the formal narrative
- Any text that is part of the actual formal story plot

When in doubt, prefer to END earlier at the last sentence of the formal narrative plot, rather than including informal supplementary content.

**CRITICAL REQUIREMENTS for end_marker**:
1. The end_marker MUST be copied EXACTLY from the text, preserving ALL characters including:
   - Spaces (leading, trailing, and between words)
   - Newline characters (\n)
   - Tab characters
   - Any special characters
2. The marker should be 10-15 words long to ensure uniqueness, but MUST extend to final punctuation mark (period, exclamation mark, question mark, etc.) of the last sentence
3. Copy the text DIRECTLY - do not paraphrase or modify it in any way

Examples:
- If the story ends with "The End.", copy exactly: "The End."
- If it ends with "and lived happily ever after.", copy exactly: "and lived happily ever after."
Do not add or skip any space.

Respond in JSON format:
{{
    "contains_end": true/false,
    "end_marker": "EXACT text snippet copied character-by-character from the text above, ending at the final punctuation mark, or null if not found",
    "reasoning": "brief explanation"
}}"""

        try:
            response = get_response(model='gemini-2.5-pro', messages=prompt, nth_generation=0)
            if not response:
                logger.warning(f"Failed to get LLM response for end detection")
                return current_pos
            
            result = extract_json(response)
            contains_end = result.get('contains_end', False)
            end_marker = result.get('end_marker')
            reasoning = result.get('reasoning', '')
            
            logger.info(f"End detection iteration {iteration + 1}: {reasoning}")
            
            if contains_end and end_marker:
                # Find the marker in the content with retry mechanism
                end_marker = end_marker.strip()
                marker_pos = content.find(end_marker, chunk_start)
                
                # Retry up to 5 times if marker not found
                retry_count = 0
                max_retries = 5
                while marker_pos == -1 and retry_count < max_retries:
                    retry_count += 1
                    logger.warning(f"End marker not found in content for '{book_title}' (attempt {retry_count}/{max_retries}): {end_marker[:50]}")
                    logger.info(f"Retrying to get a better end marker...")
                    
                    # Ask LLM again for a better marker
                    try:
                        response = get_response(model='gemini-2.5-pro', messages=prompt, nth_generation=retry_count)
                        if response:
                            result = extract_json(response)
                            new_marker = result.get('end_marker')
                            if new_marker:
                                end_marker = new_marker.strip()
                                marker_pos = content.find(end_marker, chunk_start)
                    except Exception as e:
                        logger.error(f"Error during retry {retry_count}: {e}")
                        break
                
                if marker_pos != -1:
                    end_pos = marker_pos + len(end_marker)
                    logger.info(f"Found end marker at position {end_pos} (after {retry_count} retries)")
                    return end_pos
                else:
                    logger.warning(f"End marker still not found after {max_retries} retries, using current_pos {current_pos}")
                    return current_pos  # contains_end is true, stop here even if marker not found
            
            # contains_end is false: story does not end in this chunk, move to previous chunk
            current_pos = chunk_start
            if current_pos <= start_pos:
                break
                
        except Exception as e:
            logger.error(f"Error in find_content_end iteration {iteration}: {e}")
            break
    
    return None


def extract_main_content(content: str, book_title: str) -> str:
    """
    Use LLM to identify the start and end of the main content,
    removing prefaces, introductions, afterwords, and comments.
    
    Args:
    content (str): The full book content.
    book_title (str): The title of the book.
    
    Returns:
    str: The extracted main content.
    """
    from utils import logger
    
    try:
        # Find start position
        start_pos = find_content_start(content, book_title)
        if start_pos is None:
            logger.warning(f"Could not find content start for '{book_title}', defaulting to 0")
            start_pos = 0
        
        # Find end position
        end_pos = find_content_end(content, book_title, start_pos)
        if end_pos is None:
            logger.warning(f"Could not find content end for '{book_title}', defaulting to end of content")
            end_pos = len(content)
        
        # Extract content
        extracted_content = content[start_pos:end_pos].strip()
        
        logger.info(f"Extracted main content for '{book_title}': {len(extracted_content)} chars (original: {len(content)} chars, removed: {len(content) - len(extracted_content)} chars)")
        
        return extracted_content
        
    except Exception as e:
        logger.error(f"Error in extract_main_content for '{book_title}': {e}")
        return content

def split_book(book: dict) -> dict:
    """
    Split the book content into chapters and update the book dictionary.
    
    Args:
    book (dict): The book dictionary containing 'content' key.
    
    Returns:
    dict: Updated book dictionary with 'content' as a list of chapter dictionaries.
    """
    from utils import logger
    
    content = book['content']
    
    # Regular expressions to match common chapter headings
    chapter_patterns = [
        r'\n\s*(#{1,5}\s+)?(?=.{1,50}\n)((?:Chapter|CHAPTER|Prologue|Epilogue|Afterword|Preface|Introduction|Conclusion|Appendix|Interlude|Part|PART|part|Book)|#{1,6})\s+(?:\d+|[IVXLCDM]+|(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty|Thirty|Forty|Fifty|Sixty|Seventy|Eighty|Ninety|Hundred))\.?\s*\n',

        r'\n\s*(#{1,5}\s+)?(?=.{1,40}\n)((?:Chapter|CHAPTER|Prologue|Epilogue|Afterword|Preface|Introduction|Conclusion|Appendix|Interlude|Part|PART|part))\s+.*\n',


        r'\n\s*(?=.{1,50}\n)(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty|Thirty|Forty|Fifty|Sixty|Seventy|Eighty|Ninety|Hundred)(?:\s+(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine))?\s*\n',

        r'\n\s*(#{1,5}\s+)?(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX|XXI|XXII|XXIII|XXIV|XXV|XXVI|XXVII|XXVIII|XXIX|XXX|XXXI|XXXII|XXXIII|XXXIV|XXXV|XXXVI|XXXVII|XXXVIII|XXXIX|XL|XLI|XLII|XLIII|XLIV|XLV|XLVI|XLVII|XLVIII|XLIX|L|LI|LII|LIII|LIV|LV|LVI|LVII|LVIII|LIX|LX|LXI|LXII|LXIII|LXIV|LXV|LXVI|LXVII|LXVIII|LXIX|LXX|LXXI|LXXII|LXXIII|LXXIV|LXXV|LXXVI|LXXVII|LXXVIII|LXXIX|LXXX|LXXXI|LXXXII|LXXXIII|LXXXIV|LXXXV|LXXXVI|LXXXVII|LXXXVIII|LXXXIX|XC|XCI|XCII|XCIII|XCIV|XCV|XCVI|XCVII|XCVIII|XCIX|C)\s*\n',


        r'\n\s*(?=.{1,50}\n)\d+\s*\n',

        r'\n\s*Chapter\s+(\d+)\.?\s*\n',
    ]
    
    chapter_regex = '|'.join(f'({pattern})' for pattern in chapter_patterns)
    chapter_pattern = re.compile(chapter_regex, re.IGNORECASE)
    
    # Find all chapter headings
    chapters = list(chapter_pattern.finditer(content))

    # Process chapters given the boundaries
    chapter_splits = []
    for i, match in enumerate(chapters):
        start = match.start()
        end = chapters[i+1].start() if i+1 < len(chapters) else len(content)

        if i == 0:
            # add the content before the first chapter
            chapter_content = content[:match.start()].strip()
            chapter_title = None
            chapter_splits.append({"title": chapter_title, "content": chapter_content})

        chapter_title = match.group(0).strip()
        chapter_content = content[start:end].strip()
        chapter_splits.append({"title": chapter_title, "content": chapter_content})
    
    for split in chapter_splits:
        logger.debug('===\n' + split['content'][:10])
        from utils import num_tokens_from_string
        logger.debug(f'Num tokens: {num_tokens_from_string(split["content"])}')

    
    # now merge chapter_splits. If a split < 1000 char, merge it with the NEXT split. 
    merged_chapter_splits = []
    chunk = {'title': '', 'content': ''}
    for split in chapter_splits:
        if not chunk['title']:
            chunk['title'] = split['title']
        chunk['content'] += ('' if not chunk['content'] else '\n') + split['content']
        if len(chunk['content']) >= 2000:
            merged_chapter_splits.append(chunk)
            chunk = {'title': '', 'content': ''}
    if chunk['content']:
        merged_chapter_splits.append(chunk)
    chapter_splits = merged_chapter_splits
    
    for split in chapter_splits:
        logger.debug('===\n' + split['content'][:10])
        from utils import num_tokens_from_string
        logger.debug(f'Num tokens: {num_tokens_from_string(split["content"])}')

    logger.debug(f'Splitting {book["title"]} into {len(chapter_splits)} chapters')
    

    # If we have successfully split the book
    if len(chapter_splits) > 5:
        global count_split_success
        count_split_success += 1
        
        # Clean each chapter content
        for chapter in chapter_splits:
            chapter['content'] = chapter['content'].replace('\n', ' ').replace('E b d  E - B o o k s D i r e c t o r y . c o m', ' ').replace('E b d E - B o o k s D i r e c t o r y . c o m', ' ')
            
            # Replace consecutive spaces until none remain
            while '  ' in chapter['content']:
                chapter['content'] = chapter['content'].replace('  ', ' ')
        
        # Extract main content: find start from beginning, end from the end
        logger.info(f"Extracting main content for '{book.get('title', 'Unknown')}'")
        
        # Find the first chapter that contains actual story content (start)
        first_content_chapter = 0
        first_chapter_start_pos = 0
        
        for i in range(min(3, len(chapter_splits))):  # Check first 3 chapters
            logger.info(f"Checking chapter {i+1} for story start...")
            try:
                start_pos = find_content_start(chapter_splits[i]['content'], book.get('title', 'Unknown'))
                if start_pos is not None:
                    # contains_start=true: story starts in this chapter (pos 0 is also valid)
                    first_content_chapter = i
                    first_chapter_start_pos = start_pos
                    logger.info(f"Found story start in chapter {i+1} at position {start_pos}")
                    break
                else:
                    # contains_start=false: story does not start in this chapter, try next
                    logger.info(f"Story start not found in chapter {i+1}, trying next chapter...")
            except Exception as e:
                logger.error(f"Error finding start in chapter {i+1}: {e}")
        
        # Find the last chapter that contains actual story content (end)
        last_content_chapter = len(chapter_splits) - 1
        last_chapter_end_pos = len(chapter_splits[-1]['content'])
        
        for i in range(len(chapter_splits) - 1, max(len(chapter_splits) - 4, -1), -1):  # Check last 3 chapters
            logger.info(f"Checking chapter {i+1} for story end...")
            try:
                end_pos = find_content_end(chapter_splits[i]['content'], book.get('title', 'Unknown'), 0)
                if end_pos is not None:
                    # contains_end=true: story ends in this chapter (len(content) is also valid)
                    last_content_chapter = i
                    last_chapter_end_pos = end_pos
                    logger.info(f"Found story end in chapter {i+1} at position {end_pos}")
                    break
                else:
                    # contains_end=false: story does not end in this chapter, try previous
                    logger.info(f"Story end not found in chapter {i+1}, trying previous chapter...")
            except Exception as e:
                logger.error(f"Error finding end in chapter {i+1}: {e}")
        
        # Build cleaned chapters list
        cleaned_chapters = []
        
        for i in range(first_content_chapter, last_content_chapter + 1):
            chapter = chapter_splits[i]
            
            if i == first_content_chapter and i == last_content_chapter:
                # Single chapter contains both start and end
                content = chapter['content'][first_chapter_start_pos:last_chapter_end_pos].strip()
            elif i == first_content_chapter:
                # First chapter, trim the start
                content = chapter['content'][first_chapter_start_pos:].strip()
            elif i == last_content_chapter:
                # Last chapter, trim the end
                content = chapter['content'][:last_chapter_end_pos].strip()
            else:
                # Middle chapters, keep full content
                content = chapter['content'].strip()
            
            if content:  # Only add if there's content left
                cleaned_chapters.append({
                    'title': chapter['title'],
                    'content': content
                })
        
        logger.info(f"Extracted {len(cleaned_chapters)} chapters (from chapter {first_content_chapter+1} to {last_content_chapter+1})")
        return cleaned_chapters
    else:
        # If chapter split failed, extract main content from the whole book
        logger.info(f"Chapter split failed for '{book.get('title', 'Unknown')}', extracting main content from whole book")
        
        global count_split_failed
        count_split_failed += 1
        
        # Extract main content, removing prefaces and afterwords
        content = extract_main_content(content, book.get('title', 'Unknown'))
        
        # Split the content into chunks of 8000 characters
        return content
    
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Split raw books into chapter/main-content segments.')
    parser.add_argument(
        '--input',
        type=str,
        default='dataset/original_books_from_gutenberg.jsonl',
        help='Input JSONL file containing raw books',
    )
    args = parser.parse_args()

    with jsonlines.open(args.input, mode='r') as reader:
        books_data = list(reader) 

    # Process all books
    split_books = [split_book(book) for book in books_data]

    # Print the count 
    print(f"Split {count_split_success} books, failed {count_split_failed} books")
    # Update books_data with the processed books
    books_data = split_books

    print(f"Processed {len(books_data)} books, splitting their content into chapters.")

    