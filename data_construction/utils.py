import pdb 
import os
import re 
import random 
import openai
import json
import logging
import time  
import jsonlines 
import io
import pickle
import random
import tiktoken
import __main__
from typing import Dict, List

with open('config.json', 'r') as f:
	config = json.load(f)

streaming = False

# ---------------------------------------------------------------------------
# Book-level prompt style override
# When set, all prompt-building functions use this style instead of randomly
# choosing one per sample.  Call ``set_book_prompt_style()`` at the start of
# each book / snapshot and ``clear_book_prompt_style()`` when done.
# ---------------------------------------------------------------------------
_book_prompt_style: str | None = None


def set_book_prompt_style(styles=None):
	"""Randomly pick a style and lock it for the current book/snapshot.

	All subsequent calls to ``build_diverse_task_prompt``,
	``_build_rich_next_character_prompt``, and
	``_build_rich_interaction_generation_prompt`` will use this style
	instead of sampling independently.
	"""
	global _book_prompt_style
	styles = styles or ['natural'] * 40 + ['='] * 30 + ['#'] * 20 + ['*'] * 10
	_book_prompt_style = random.choice(styles)
	return _book_prompt_style


def clear_book_prompt_style():
	"""Remove the book-level style lock so each sample picks its own style."""
	global _book_prompt_style
	_book_prompt_style = None


def _resolve_prompt_style(styles=None):
	"""Return the book-level style if set, otherwise randomly pick one."""
	if _book_prompt_style is not None:
		return _book_prompt_style
	styles = styles or ['natural'] * 40 + ['='] * 30 + ['#'] * 20 + ['*'] * 10
	return random.choice(styles)

def setup_logger(name, log_file, level=logging.INFO, console_level=None, quiet=False):
	"""Setup logger with separate file and console log levels.
	
	Args:
		name: Logger name
		log_file: Path to log file
		level: File log level (default: INFO)
		console_level: Console log level (default: same as level). Set to WARNING to hide INFO logs from terminal.
		quiet: If True, disable console output
	
	Returns:
		Configured logger instance
	"""
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

	if logger.hasHandlers():
		logger.handlers.clear()

	# File handler - captures all logs at specified level
	file_handler = logging.FileHandler(log_file, encoding='utf-8')
	file_handler.setLevel(level)
	file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	file_handler.setFormatter(file_formatter)
	logger.addHandler(file_handler)

	# Console handler - can have different level than file
	if not quiet:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(console_level if console_level is not None else level)
		console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
		console_handler.setFormatter(console_formatter)
		logger.addHandler(console_handler)

	return logger

logger = setup_logger(__name__, 'data_construction/utils.log', level=logging.INFO, console_level=logging.WARNING, quiet=False)

from contextlib import contextmanager
import tempfile
@contextmanager
def _tempfile(dir=None,*args, **kws):
	""" Context for temporary file.
	Will find a free temporary filename upon entering
	and will try to delete the file on leaving
	Parameters
	----------
	suffix : string
		optional file suffix
	dir : string
		directory to create temp file in, will be created if doesn't exist
	"""
	if dir is not None:
		os.makedirs(dir, exist_ok=True)
		
	fd, name = tempfile.mkstemp(dir=dir, *args, **kws)
	os.close(fd)
	try:
		yield name
	finally:
		try:
			os.remove(name)
		except OSError as e:
			if e.errno == 2:
				pass
			else:
				raise e
			
@contextmanager
def open_atomic(filepath, *args, **kwargs):
	""" Open temporary file object that atomically moves to destination upon
	exiting.
	Allows reading and writing to and from the same filename.
	Parameters
	----------
	filepath : string
		the file path to be opened
	fsync : bool
		whether to force write the file to disk
	kwargs : mixed
		Any valid keyword arguments for :code:`open`
	"""
	fsync = kwargs.pop('fsync', False)

	original_permissions = os.stat(filepath).st_mode if os.path.exists(filepath) else None 

	with _tempfile(dir=os.path.join(os.path.dirname(filepath), 'temp')) as tmppath:
		with open(tmppath, *args, **kwargs) as f:
			yield f
			if fsync:
				f.flush()
				os.fsync(f.fileno())
		os.rename(tmppath, filepath)
		if original_permissions is not None:
			os.chmod(filepath, original_permissions)

import datetime
def convert_to_timestamp(time_str: str):
	return time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple())

def safe_pickle_dump(obj, fname):
	"""
	prevents a case where one process could be writing a pickle file
	while another process is reading it, causing a crash. the solution
	is to write the pickle file to a temporary file and then move it.
	"""
	with open_atomic(fname, 'wb') as f:
		pickle.dump(obj, f, -1) # -1 specifies highest binary protocol


ERROR_SIGN = '[ERROR]'

cache_path = '.cache.pkl'
cache_sign = True
cache = None
reload_cache = False

def set_cache_path(new_cache_path):
	global cache_path
	cache_path = new_cache_path
	global reload_cache
	reload_cache = True

def cached(func):
	def wrapper(*args, **kwargs):		
		# extract_from_chunk 
		if func.__name__ == 'extract_from_chunk':
			key = ( func.__name__, args[0]['title'], args[1]) 
		else:
			key = ( func.__name__, str(args), str(kwargs.items())) 

		global cache
		global reload_cache

		if reload_cache:
			cache = None # to reload
			reload_cache = False

		if cache == None:
			if not os.path.exists(cache_path):
				cache = {}
			else:
				try:
					cache = pickle.load(open(cache_path, 'rb'))  
				except Exception as e:
					# logger.info cache_path and throw error
					logger.error(f'Error loading cache from {cache_path}')
					cache = {}

		if (cache_sign and key in cache) and not (cache[key] is None):
			return cache[key]
		else:		
			result = func(*args, **kwargs)
			if result != None:
				cache[key] = result
				safe_pickle_dump(cache, cache_path)
			return result

	return wrapper

enc = tiktoken.get_encoding("cl100k_base")  # Claude uses cl100k_base encoding

def encode(text):
	return enc.encode(text)

def decode(tokens):
	return enc.decode(tokens)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
	encoding = tiktoken.get_encoding(encoding_name)
	num_tokens = len(encoding.encode(string))
	logger.info(f"Number of tokens: {num_tokens}")
	return num_tokens

@cached
def get_response(model, messages, nth_generation=0, max_retries=3, **kwargs):
	# if messages is str
	if isinstance(messages, str):
		messages = [{"role": "user", "content": messages}]

	response = None  # Initialize response to avoid UnboundLocalError
	
	for retry_count in range(max_retries):
		try:
			import openai 
			client = openai.OpenAI(api_key=config['api_key'] or 'EMPTY', base_url=config['base_url'], timeout=720, default_headers=config.get('extra_headers') or None)
			# No provider key -> gateway-side auth (e.g. BYOK). Omit the SDK's
			# placeholder Authorization header so it can't override stored keys.
			request_headers = None if config['api_key'] else {'Authorization': openai.Omit()}

			max_tokens = 65536

			# gpt-5.2 and o-series models use max_completion_tokens instead of max_tokens
			# (gateways may namespace models as "<provider>/<model>"; strip to the bare name first)
			bare_model = model.rsplit('/', 1)[-1]
			use_completion_tokens = bare_model.startswith('o') or bare_model.startswith('gpt-5')
			token_param = 'max_completion_tokens' if use_completion_tokens else 'max_tokens'

			if streaming:
				stream = client.chat.completions.create(
					model=model,
					messages=messages,
					stream=True,
					**{token_param: max_tokens},
					extra_headers=request_headers,
					temperature=0 if nth_generation == 0 else 1,
					timeout=720
				)

				response = ""
				for chunk in stream:
					try:
						if chunk.choices[0].delta.content is not None:
							response += chunk.choices[0].delta.content
					except:
						if len(response) == 0:
							return None

						if len(chunk.choices) == 0 and response.strip()[-1] == '}':
							break 
			else:
				completion = client.chat.completions.create(
					model=model,
					messages=messages,
					**{token_param: max_tokens},
					extra_headers=request_headers,
					temperature=0 if nth_generation == 0 else 1,
					timeout=720
				)
				response = completion.choices[0].message.content
			
			return response

		except openai.RateLimitError as e:
			error_msg = str(e)
			# Extract retry_after from error message if available
			retry_after_match = re.search(r'retry_after[\'"]?\s*:\s*(\d+)', error_msg)
			wait_time = int(retry_after_match.group(1)) if retry_after_match else 60
			logger.warning(f"Rate limit exceeded (attempt {retry_count + 1}/{max_retries}). Waiting {wait_time}s before retry...")
			if retry_count < max_retries - 1:
				time.sleep(wait_time)
				continue
			else:
				logger.error(f"Rate limit exceeded after {max_retries} attempts")
				return None

		except openai.InternalServerError as e:
			error_msg = str(e)
			# Check if it's a WAF block (Tencent Cloud WAF)
			if 'waf.tencent.com' in error_msg or '501page.html' in error_msg:
				logger.warning(f"WAF blocked request (attempt {retry_count + 1}/{max_retries}). Waiting before retry...")
				if retry_count < max_retries - 1:
					wait_time = 30  # 30s
					time.sleep(wait_time)
					continue
				else:
					logger.error(f"WAF blocked request after {max_retries} attempts")
					logger.error(f"Number of input tokens: {num_tokens_from_string(messages[0]['content'])}")
					return None
			else:
				# Other internal server errors
				logger.error(f"InternalServerError: {error_msg}")
				if retry_count < max_retries - 1:
					time.sleep(2)
					continue
				else:
					return None
					
		except Exception as e:
			import traceback 
			logger.error(f'Prompt (first 500 chars): {str(messages)[:500]}')
			logger.error(f"Error in get_response: {str(e)}")

			# Only try to print response if it was assigned
			if response is not None:
				try:
					if hasattr(response, 'text'):
						logger.error(f"Response: {response.text}")
					else:
						logger.error(f"Response: {response}")
				except Exception as print_error:
					logger.error(f"Could not print response: {print_error}")
			
			logger.error(f"Number of input tokens: {num_tokens_from_string(messages[0]['content'])}")

			traceback.print_exc()
			
			# Retry on certain errors
			if retry_count < max_retries - 1:
				logger.info(f"Retrying... (attempt {retry_count + 2}/{max_retries})")
				time.sleep(2)
				continue
			else:
				return None
	
	return None
	
def lang_detect(text):
	import re
	def count_chinese_characters(text):
		chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
		return len(chinese_chars)
			
	if count_chinese_characters(text) > len(text) * 0.05:
		lang = 'zh'
	else:
		lang = 'en'
	return lang
	

def remove_inner_thoughts(content: str) -> str:
	cleaned_content = re.sub(r'\[.*?\]', '', content)

	cleaned_content = '\n'.join(line.strip() for line in cleaned_content.split('\n'))
	
	cleaned_content = re.sub(r'\n+', '\n', cleaned_content)
	
	return cleaned_content.strip()

def add_speaker_name(content: str, speaker: str) -> str:
	# Check if the content already contains a speaker prefix at the beginning of any line
	if any(line.strip().startswith(f"{speaker}:") or line.strip().startswith(f"{speaker}:") for line in content.split('\n')):
		return content
	
	# Add the speaker name at the beginning
	return f"{speaker}: {content}"


def load_json(file_path):
	with open(file_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	return data


# Mapping from style category to concrete format templates
_HEADING_TEMPLATES = {
	'plain': ["{title}:"],
	'=': ["==={title}===", "=={title}==", "={title}="],
	'#': ["#{title}", "# {title}", "## {title}", "### {title}"],
	'*': ["**{title}**", "*{title}*", "***{title}***"],
}


def _pick_heading_template(style: str) -> str:
	"""Pick a concrete heading template for the given style category.
	Call once per prompt so that all sections share the same format."""
	templates = _HEADING_TEMPLATES.get(style)
	if templates:
		return random.choice(templates)
	return "{title}"


def _prompt_heading(title: str, template: str) -> str:
	"""Apply a pre-selected heading template to a section title."""
	return template.format(title=title)



def _prompt_value_to_str(value) -> str:
	if value is None:
		return ""
	if isinstance(value, (dict, list)):
		return json.dumps(value, ensure_ascii=False, indent=2)
	return str(value)



def _bulletize(items) -> str:
	return "\n".join(f"- {item}" for item in items if item)



def _pick_section_title(title, aliases=None):
	aliases = aliases or {}
	choices = aliases.get(title, [title])
	return random.choice(choices)



def _render_prompt_sections(sections, style: str = None, section_aliases=None) -> str:
	style = style or random.choice(['plain'] * 40 + ['='] * 30 + ['#'] * 20 + ['*'] * 10)
	heading_template = _pick_heading_template(style)
	section_aliases = section_aliases or {}
	blocks = []
	for title, content in sections:
		content_str = _prompt_value_to_str(content).strip()
		if not content_str:
			continue
		final_title = _pick_section_title(title, section_aliases)
		blocks.append(f"{_prompt_heading(final_title, heading_template)}\n{content_str}")
	return "\n\n".join(blocks)



def _build_natural_prompt(opening, task_line, input_lines, output_lines, rules, context_sections, section_aliases=None):
	section_aliases = section_aliases or {}
	sections = [
		("Task", task_line),
		("Output", _bulletize(output_lines)),
	]
	if rules:
		sections.append(("Rules", _bulletize(rules)))
	if input_lines:
		sections.append(("Inputs", _bulletize(input_lines)))
	for title, content in context_sections or []:
		content_str = _prompt_value_to_str(content).strip()
		if not content_str:
			continue
		sections.append((title, content_str))
	section_text = _render_prompt_sections(sections, style='plain', section_aliases=section_aliases)
	return f"{opening}\n\n{section_text}".strip()



def build_diverse_task_prompt(role_variants, task_variants, input_specs, output_specs, rules=None, context_sections=None, section_aliases=None, styles=None):
	opening = random.choice(role_variants).strip()
	task_line = random.choice(task_variants).strip()
	input_lines = [f"{name}: {desc}" for name, desc in input_specs]
	output_lines = [f"{name}: {desc}" for name, desc in output_specs]
	section_aliases = section_aliases or {}
	sections = [
		("Task", task_line),
		("Output", _bulletize(output_lines)),
	]
	if rules:
		sections.append(("Rules", _bulletize(rules)))
	if input_lines:
		sections.append(("Inputs", _bulletize(input_lines)))
	if context_sections:
		for title, content in context_sections:
			content_str = _prompt_value_to_str(content).strip()
			if not content_str:
				continue
			sections.append((title, content_str))
	section_text = _render_prompt_sections(sections, style='plain', section_aliases=section_aliases)
	return f"{opening}\n\n{section_text}".strip()



def _format_prompt_block(value, default='(None)'):
	if value is None:
		return default
	if isinstance(value, str):
		return value if value.strip() else default
	if isinstance(value, (dict, list)):
		text = json.dumps(value, ensure_ascii=False, indent=2)
		return text if text.strip() else default
	text = str(value)
	return text if text.strip() else default



def _build_rich_next_character_prompt(global_card, location_name, location_card, scenario, char_descs_in_scene, prior_interactions=None):
	all_characters = list(char_descs_in_scene.keys())
	allowed_names = all_characters + ['Environment', '<SCENE_END>']
	world_state_text = (
		f"Current location: \"{location_name}\"\n\n"
		f"Global world state:\n{_format_prompt_block(global_card)}\n\n"
		f"Location state:\n{_format_prompt_block(location_card)}"
	)
	characters_text = _format_prompt_block(char_descs_in_scene)
	prior_text = _format_prompt_block(prior_interactions)

	return (
		"Task: choose who acts next RIGHT NOW in the current scene.\n"
		"Output only one JSON list and nothing else.\n\n"
		f"Allowed outputs: {allowed_names}.\n"
		"- Use one present character name for a normal turn.\n"
		"- Use ['Environment'] only for a non-character environmental beat.\n"
		"- Use ['<SCENE_END>'] only if the scene should end now.\n"
		"- Output multiple characters together only when the very next beat is one indivisible shared interaction that must be realized by those characters together right now.\n"
		"- Do not output a character group just because multiple characters are present, aligned, nearby, or likely to act one after another.\n"
		"- If one character can naturally act first and the others can respond afterward, choose only that one character.\n"
		"- A multi-character output should include only the characters who must jointly produce the same immediate beat, with no extra passengers.\n\n"
		"Decision priorities:\n"
		"1. Continue directly from the latest visible interaction cue.\n"
		"2. Keep the scene moving forward; do not restart the scene or repeat an already completed beat.\n"
		"3. Use character motivations and the current world state as support for what most naturally happens next.\n"
		"4. When unsure, prefer a single-character turn over a group turn unless the next beat genuinely requires simultaneous or jointly authored participation.\n"
		"5. Choose a multi-character list only for cases like a joint physical action, a jointly delivered line, or a tightly coupled shared reaction that belongs in one beat rather than split turns.\n"
		"6. Do not choose the same role as the current last speaker; avoid consecutive turns by the same character or actor group unless no other continuation is plausible.\n\n"
		"Timeline:\n"
		"- The scene scenario and character states are from scene start.\n"
		"- Prior interactions happened after scene start.\n"
		"- The current world state is the result of those prior interactions.\n"
		"- The live conversation after this prompt continues immediately after that point.\n\n"
		"Conversation protocol:\n"
		"- Each user message is an interaction in the format [\"CharA\", \"CharB\", ...]: content.\n"
		"- The content may contain [...] for inner thoughts, plain text for speech, and (...) for visible actions.\n"
		"- A multi-character list means one shared interaction beat, not separate consecutive turns.\n"
		"- Return raw JSON only. No markdown. No explanation.\n\n"
		f"## Characters At Scene Start\n{characters_text}\n\n"
		f"## Scene Scenario\n{_format_prompt_block(scenario)}\n\n"
		f"## Current World State\n{world_state_text}\n\n"
		f"## Prior Interactions Before Current World State\n{prior_text}\n\n"
		"Final reminder before the live segment starts: choose the next actor only, and identify the current last speaker from the full interaction history available at this moment, including the prior interactions above and any live conversation after segment start; do not select the same role as that last speaker for the next turn."
	).strip()



def _build_rich_interaction_generation_prompt(book_name, acting_characters, location_name, global_card, location_card, scenario, actor_states=None, other_char_descs=None, prior_interactions=None, is_environment=False):
	actor_label = ', '.join(acting_characters)
	world_state_text = (
		f"Book: \"{book_name}\"\n\n"
		f"Current location: \"{location_name}\"\n\n"
		f"Global world state:\n{_format_prompt_block(global_card)}\n\n"
		f"Location state:\n{_format_prompt_block(location_card)}"
	)
	prior_text = _format_prompt_block(prior_interactions)

	if is_environment:
		major_characters = [c for c in (other_char_descs or {}).keys() if c != 'Environment']
		return (
			"Task: write exactly one next environmental beat RIGHT NOW.\n"
			"Continue directly from the latest visible interaction cue. Do not restart the scene. Do not repeat or paraphrase an already completed beat.\n\n"
			"Requirements:\n"
			"- Treat this beat as happening immediately after the prior interactions below.\n"
			"- Write exactly one short environmental turn.\n"
			"- Focus on atmosphere, background movement, physical changes, crowd reaction, sound, weather, or setting consequences.\n"
			f"- Do not take over the deliberate dialogue or private thoughts of the main characters ({major_characters}).\n"
			"- Use the current world state only as support; follow the interaction flow most closely.\n\n"
			"Conversation protocol:\n"
			"- The first user message is only '===Segment Start===' and should not be answered literally.\n"
			"- Every later user message is a new interaction that happens after the snapshot below.\n\n"
			f"## Scene Scenario\n{_format_prompt_block(scenario)}\n\n"
			f"## Current World State\n{world_state_text}\n\n"
			f"## Prior Interactions Before Current World State\n{prior_text}\n\n"
			"Final reminder before the live segment starts: you are roleplaying the environment layer. Do not repeat any interaction beat already covered in the prior interactions above or in the live conversation after segment start, and do not turn this beat into a main-character dialogue turn."
		).strip()

	actor_state_text = _format_prompt_block(actor_states)
	other_characters_text = _format_prompt_block(other_char_descs)
	multi_char = len(acting_characters) > 1

	if multi_char:
		role_block = (
			f"Task: roleplay {actor_label} and write exactly one next shared interaction beat RIGHT NOW.\n"
			f"You are {actor_label} now. Continue directly from the latest visible interaction cue. Do not restart the scene. Do not repeat or paraphrase an already completed beat.\n\n"
			"Shared-turn rules:\n"
			"- Treat the turn as happening immediately after the prior interactions below.\n"
			"- Only write one truly shared beat.\n"
			"- This shared beat is one local moment jointly realized by the acting group, not a bundle of separate back-to-back turns.\n"
			"- The group should do or say one thing together: for example a joint action, a jointly delivered line, or one tightly coupled shared reaction unfolding in the same moment.\n"
			"- Do not split into separate mini-turns for each character, do not serialize the group into first X then Y then Z, and do not give unrelated contributions from different members in the same output.\n"
			"- Include only material that belongs to this one shared moment. If a member's contribution would naturally happen later as a follow-up, leave it out.\n"
			"- Thoughts from members of the acting group may be used inside this shared turn, but only when they support the same shared moment.\n\n"
		)
	else:
		role_block = (
			f"Task: roleplay {actor_label} and write exactly one next interaction turn RIGHT NOW.\n"
			f"You are {actor_label} now. Continue directly from the latest visible interaction cue. Do not restart the scene. Do not repeat or paraphrase an already completed beat.\n\n"
		)

	return (
		role_block
		+ "Priority rules:\n"
		+ "1. Stay in character.\n"
		+ "2. React or act from the latest visible cue, not from the beginning of the scene.\n"
		+ "3. Keep the whole scene logically continuous. The new turn must advance or react within the same ongoing scene.\n"
		+ "4. Do not restate, replay, or slightly reword an earlier beat.\n"
		+ "5. Use world state and character state only as support for continuity; follow the interaction flow most closely.\n"
		+ "6. Keep the turn short and local. No summary. No jump ahead.\n\n"
		+ "Output format:\n"
		+ "- Write exactly one full turn.\n"
		+ "- Start with a real inner-thought block in square brackets.\n"
		+ "- Put spoken dialogue in plain text with no speaker label.\n"
		+ "- Do NOT put spoken dialogue inside square brackets, and do NOT wrap speech-only text in parentheses such as [Who is that?] (I say).\n"
		+ "- Put visible physical actions in parentheses.\n"
		+ "- These elements can be interleaved naturally.\n\n"
		+ "Conversation protocol:\n"
		+ "- The first user message is only '===Segment Start===' and should not be answered literally.\n"
		+ "- Every later user message is a new visible interaction that happens after the snapshot below.\n"
		+ "- Each user message uses the format [\"CharA\", \"CharB\", ...]: content.\n"
		+ "- Other characters' private thoughts are already removed unless the acting group overlaps with them.\n\n"
		+ f"## Your Character State\n{actor_state_text}\n\n"
		+ (f"## Other Characters\n{other_characters_text}\n\n" if other_characters_text != '(None)' else "")
		+ f"## Scene Scenario\n{_format_prompt_block(scenario)}\n\n"
		+ f"## Current World State\n{world_state_text}\n\n"
		+ f"## Prior Interactions Before Current World State\n{prior_text}\n\n"
		+ f"Final reminder before the live segment starts: you are roleplaying {actor_label}. Do not repeat any interaction beat already covered in the prior interactions above or in the live conversation after segment start, including anything this same role has already done or said there; continue with one new turn only."
	).strip()

def get_scene_cast_system_prompt():
	return build_diverse_task_prompt(
		role_variants=[
			"You are the scene-cast planning module for a story simulation. Your job is to decide whether another scene should happen and, if so, which characters should be in that scene's cast.",
			"You are the director-level scene casting planner. Decide whether the story continues and who belongs in the full cast of the next scene.",
			"You are responsible for planning the cast lineup of the next scene based on the current world and character states.",
			"You are the first-stage scene planner. Before the next scene begins, decide whether it should exist and which characters should be included in its cast.",
			"You are the world-planning module that determines whether the story should continue into another scene and, if so, which characters should make up that next scene's cast.",
		],
		task_variants=[
			"Determine whether a next scene exists, and if it does, choose the full set of characters who should participate in that scene.",
			"Use the current story state to decide whether the story continues and which characters belong in the next scene's cast.",
			"Judge whether the narrative naturally leads to another scene, and if so, select its cast before the scene starts.",
			"Plan the immediate next scene by choosing the most plausible cast based on continuity and the characters' current visible states.",
			"From the current world state and character descriptions, decide the next scene's cast, or conclude that the story ends here.",
		],
		input_specs=[
			("Global World State", "The current high-level state of the whole story world (reflects all events up to now), including stable world facts and major ongoing conditions."),
			("All Characters (Short Description Only)", "A snapshot of each character's current visible state as of the end of the last scene, before the next scene begins."),
			("Previous Scene Scenario", "The scenario/dramatic setup of the scene that just ended. Use this to understand what just happened and ensure the next scene follows naturally. This may be '(None)' if this is the first scene."),
			("Previous Scene Interactions", "The full interaction history of the scene that just ended, showing what the characters said and did. Use this to maintain narrative continuity. This may be '(None)' if this is the first scene."),
		],
		output_specs=[
			("has_next_scene", "Boolean. Output false only if there is no subsequent scene."),
			("involved_characters", "List of the characters participating in the next scene. Include this only when has_next_scene is true."),
		],
		rules=[
			"Base the decision on continuity and plausibility rather than novelty alone.",
			"Use the current world state and each character's latest visible description to decide which characters belong in the next scene's cast and why.",
			"Pay close attention to the previous scene's scenario and interactions to ensure the next scene follows naturally from what just happened. The cast should reflect the narrative momentum and unresolved threads from the previous scene.",
			"This is a scene-level casting decision made before the scene starts, not a turn-by-turn next-actor prediction inside the scene.",
			"Do not choose a cast that clearly contradicts the current world state or the characters' latest visible states.",
			"If has_next_scene is false, do not include involved_characters.",
			"Return JSON only.",
		],
		section_aliases={
			"Inputs": ["Inputs", "Available Inputs", "What Each Input Means"],
			"Output": ["Output", "Expected Output", "Required JSON Fields"],
			"Rules": ["Rules", "Requirements", "Planning Principles"],
		},
	)



def get_scene_location_scenario_system_prompt():
	return build_diverse_task_prompt(
		role_variants=[
			"You are the second-stage scene planner for a story simulation. Your job is to place the already-selected characters into a concrete next scene.",
			"You are the location-and-scenario planning module. Given the selected cast, decide where the next scene should happen and what its dramatic setup is.",
			"You are the director-level scene setup planner. You receive the chosen characters and must decide the next location and scenario.",
			"You are responsible for turning a selected group of characters into a concrete next scene by choosing the location and writing the scenario.",
			"You are the world-planning module that finalizes the next scene after the participating characters have already been chosen.",
		],
		task_variants=[
			"Choose the most plausible location for the selected characters and generate the next scene's scenario.",
			"Use the current world state, location overviews, and the selected characters' current descriptions to decide where the next scene happens and what unfolds there.",
			"Finalize the next scene by selecting a location and writing a concise but usable scenario.",
			"Given the selected cast for the next scene, decide where they should meet and describe the dramatic setup.",
			"Plan the concrete setup of the next scene by choosing one location and generating the scenario for the selected characters.",
		],
		input_specs=[
			("Global World State", "The current high-level state of the whole story world (reflects all events up to now), including stable world facts and major ongoing conditions."),
			("All Location Descriptions", "A map of locations to short current descriptions (updated after the most recent scene); use this to determine where the next-scene characters could plausibly go next."),
			("Characters Who Will Appear In The Next Scene (Short Description Only)", "The characters who will appear in the next scene, together with their visible current states."),
			("Previous Scene Scenario", "The scenario/dramatic setup of the scene that just ended. Use this to understand the narrative context and ensure the next scene continues naturally. This may be '(None)' if this is the first scene."),
			("Previous Scene Interactions", "The full interaction history of the scene that just ended, showing what the characters said and did. Use this to maintain narrative continuity and build the next scenario as a natural follow-up. This may be '(None)' if this is the first scene."),
		],
		output_specs=[
			("location", "The next scene's location. Output null only if there is no valid next-scene location to assign."),
			("scenario", "A concise but usable dramatic foundation for the next scene. It should establish the immediate setup, atmosphere, and enough background for downstream actors to perform the scene. Even if location is null, still output a scenario whenever the next scene itself is valid."),
		],
		rules=[
			"Base the decision on continuity and plausibility rather than novelty alone.",
			"Use the selected characters' current visible descriptions and the global world state to justify why this location makes sense now.",
			"Pay close attention to the previous scene's scenario and interactions. The new scenario should follow naturally from what just happened — continuing unresolved conflicts, reacting to recent events, or advancing the narrative arc established in the previous scene.",
			"The scenario should describe what kind of situation is unfolding, not script the dialogue itself.",
			"Keep scenario concise but informative enough for downstream acting.",
			"If there is no valid next scene to set up, output location=null and scenario=null.",
			"If the next scene exists but the exact location is unknown or unspecified, you may output location=null while still providing a concrete scenario.",
			"Return JSON only.",
		],
		section_aliases={
			"Inputs": ["Inputs", "Available Inputs", "What Each Input Means"],
			"Output": ["Output", "Expected Output", "Required JSON Fields"],
			"Rules": ["Rules", "Requirements", "Planning Principles"],
		},
	)


def get_scene_proposal_system_prompt():
	return get_scene_location_scenario_system_prompt()


def get_scene_character_selection_system_prompt():
	return get_scene_cast_system_prompt()


def get_next_character_system_prompt(global_card, location_name, location_card, scenario, char_descs_in_scene, prior_interactions=None):
	return _build_rich_next_character_prompt(
		global_card=global_card,
		location_name=location_name,
		location_card=location_card,
		scenario=scenario,
		char_descs_in_scene=char_descs_in_scene,
		prior_interactions=prior_interactions,
	)



def get_world_state_update_system_prompt():
	return build_diverse_task_prompt(
		role_variants=[
			"You are the world-state update judge for a story simulation. Your job is to decide whether an interaction causes a lasting change to the persistent world state.",
			"You are the world-state controller. After each interaction, determine whether the global or location state needs a persistent update.",
			"You are the persistent-state maintenance module. Decide whether the latest interaction warrants updating the stored world state.",
			"You are the module that tracks lasting changes to the story world. Judge whether the newest interaction should trigger a world-state revision.",
			"You are responsible for deciding when the persistent world representation must be revised based on the latest interaction.",
		],
		task_variants=[
			"Given the scene context, prior interactions, current world state, and the latest interaction, decide whether the global world state and/or the current location state must be updated.",
			"Judge whether the latest interaction causes a meaningful persistent update to world state, considering the scene background and prior context.",
			"Review the latest interaction against the scene scenario and prior history, then update the relevant world state only when lasting change occurs.",
			"Determine whether the newest interaction creates a durable change that should now be reflected in persistent world memory.",
			"Inspect the latest interaction in light of the scene context and current world state, and revise world state only when something meaningfully and persistently changes.",
		],
		input_specs=[
			("Scene Scenario", "The scene-level dramatic background (established at scene start) that frames all interactions."),
			("Prior Interactions", "All interactions that happened before the latest one. These prior interactions occurred BEFORE the current global/location world state shown below and together explain how that current state was reached."),
			("Global World State", "The current persistent global state, already updated after the prior interactions above and covering all events up to (but not including) the latest interaction."),
			("Current Location State", "The current persistent location-specific state, already updated after the prior interactions above and covering all events up to (but not including) the latest interaction."),
			("Latest Interaction", "The single newest interaction that you must judge — decide whether it causes a lasting change to the world state."),
		],
		output_specs=[
			("update_global", "Boolean. True only when the latest interaction changes the persistent global world state in a meaningful way."),
			("global_state", "The complete updated global state when update_global is true, otherwise null."),
			("update_location", "Boolean. True only when the latest interaction changes the persistent state of the current location in a meaningful way."),
			("location_state", "The complete updated location state when update_location is true, otherwise null."),
		],
		rules=[
			"Update the global world state only for broad, lasting systemic changes.",
			"Update the location state only for lasting local changes to environment, atmosphere, or important non-human entities.",
			"Do not update for transient actions that leave no persistent consequence.",
			"When updating, output the full new state rather than a diff.",
			"Be conservative: most interactions should not force a world-state update.",
			"Return JSON only.",
		],
		section_aliases={
			"Inputs": ["Inputs", "State Inputs", "Context For World-State Decisions"],
			"Output": ["Output", "Required JSON Fields", "What To Return"],
			"Rules": ["Rules", "Update Principles", "Decision Constraints"],
		},
	)



def get_interaction_generation_system_prompt(book_name, acting_characters, location_name, global_card, location_card, scenario, actor_states=None, other_char_descs=None, prior_interactions=None, is_environment=False):
	return _build_rich_interaction_generation_prompt(
		book_name=book_name,
		acting_characters=acting_characters,
		location_name=location_name,
		global_card=global_card,
		location_card=location_card,
		scenario=scenario,
		actor_states=actor_states,
		other_char_descs=other_char_descs,
		prior_interactions=prior_interactions,
		is_environment=is_environment,
	)



def get_character_state_update_system_prompt(character_name):
	return build_diverse_task_prompt(
		role_variants=[
			f'You are tracking how "{character_name}" evolves throughout a story, scene by scene. Your goal is to maintain a comprehensive, up-to-date internal state that captures who this character is and how they change over time.',
			f'You are the post-scene state manager for "{character_name}". After each completed scene, you analyze what happened and determine how the character\'s internal state — profile, hidden tracker, and description — should be updated.',
			f'You are responsible for updating "{character_name}"\'s persistent internal state after a completed scene. You must reason carefully about what changed and what accumulated.',
			f'You are the character evolution module for "{character_name}". After each scene, you decide which aspects of the character\'s state need updating and produce a structured state snapshot for use in future scenes.',
			f'You are revising "{character_name}"\'s internal state after a scene concludes. Your updates will serve as the foundation for this character\'s behavior in all subsequent scenes.',
		],
		task_variants=[
			(
				"Follow these steps in order:\n"
				"1. **Reason about dimensions**: Identify which profile dimensions are STABLE (e.g., physical description, backstory) vs. DYNAMIC (e.g., relationships, goals, mental state) for this character.\n"
				"2. **Update the Hidden Tracker**: Record events, unresolved tensions, accumulated emotional pressure, or signals from this scene that may lead to future profile changes — even if the profile itself doesn't change yet.\n"
				"3. **Decide whether to update the profile**: Only update if a meaningful, lasting change occurred (e.g., a relationship shift, a major decision, a trauma). Do NOT update for minor or transient reactions.\n"
				"4. **Write a short description**: Summarize who this character is NOW in 50–80 words, third person. Focus on identity, current situation, key relationships, and immediate condition at the end of the scene."
			),
			(
				"Perform the following tasks sequentially:\n"
				"1. **Dimension reasoning**: Briefly reason about which of this character's profile dimensions are stable vs. dynamic.\n"
				"2. **Hidden tracker update**: Update the tracker with accumulated events/signals from this scene that may eventually trigger a profile change.\n"
				"3. **Profile update decision**: Determine whether the scene justifies a persistent profile change. If yes, produce the full updated profile; if no, leave it null.\n"
				"4. **Short description**: Write a concise description (50–80 words) of the character's current identity, situation, and immediate condition after this scene."
			),
			(
				"Process the completed scene step by step:\n"
				"1. Reason about which character dimensions are likely stable vs. dynamic across the story.\n"
				"2. Update the hidden tracker: record events, emotional pressure, or unresolved tensions that may lead to future profile changes.\n"
				"3. Decide if the profile should be updated NOW — only when the scene causes a meaningful, observable, lasting change. Accumulated tracker signals combined with this scene may cross the threshold.\n"
				"4. Write a short description (50–80 words, third person) covering who the character is right now and what condition they are left in at the end of the scene."
			),
			(
				"Analyze the scene and update state in this order:\n"
				"1. **Stable vs. dynamic dimensions**: Which aspects of this character's profile are unlikely to change vs. which can evolve with events?\n"
				"2. **Hidden tracker**: Accumulate signals — experiences, interactions, emotional pressure, unresolved decisions — that haven't yet caused a visible profile change but may in the future.\n"
				"3. **Profile update**: Only update when the hidden tracker plus this scene's events cross a threshold for genuine, lasting character change. Be selective.\n"
				"4. **Description**: A brief snapshot (50–80 words) of the character's current identity, situation, and immediate condition after the scene."
			),
			(
				"Your job is to translate the completed scene into character-state updates:\n"
				"1. First, reason about which profile dimensions are stable (backstory, physical traits) vs. dynamic (relationships, goals, emotional state).\n"
				"2. Then, update the hidden tracker with events and signals that may accumulate toward future profile changes.\n"
				"3. Next, decide whether to update the profile — require meaningful, lasting change, not transient reactions.\n"
				"4. Write a short description (50–80 words) of the character as of now."
			),
		],
		input_specs=[
			("Scene Scenario", "The dramatic setup and background of the scene, established at scene start before any interactions happened."),
			("Scene Interactions", "The full scene interaction history from this character's perspective (other characters' thoughts are hidden; everything that happened during the scene). Use this as the main evidence for how the character changed by scene end."),
			("Current Profile", "The character's full profile at SCENE START, before any interactions in this scene. Compare it against the scene interactions to decide whether the end-of-scene profile should change."),
			("Hidden Tracker", "The hidden tracker at SCENE START, carried in from previous scenes before this scene's interactions unfolded. Update it using what happened in the scene."),
			("Current Motivation", "The character's active inner drive at SCENE START: their emotional state, immediate objectives, and action intentions before the scene interactions took place."),
		],
		output_specs=[
			("hidden_tracker", "Updated tracker of accumulated events/signals (experiences, emotional pressure, unresolved tensions) that may lead to future profile changes. Overwrite the old tracker with the updated version. Null if there are no signals worth tracking."),
			("profile_updated", "Boolean: true only if the scene caused a meaningful, lasting change to the character's profile. Be selective — do not update for minor or transient reactions."),
			("updated_profile", "The full updated profile as a plain text string when profile_updated is true. Include all relevant dimensions. Null if profile_updated is false."),
			("short_description", "A brief description (50–80 words, third person) of the character as of the end of this scene: current identity, key relationships, situation, and immediate goals/intentions."),
		],
		rules=[
			"Follow the step-by-step reasoning order: dimension reasoning → hidden tracker → profile decision → description.",
			"All outputs must reflect the character's state as of the END of the current scene only.",
			"Keep hidden_tracker private and internal-facing — it tracks accumulated signals, not public information.",
			"Update the profile only when the scene justifies a persistent, lasting character-level change. Accumulated tracker signals combined with this scene may cross the threshold.",
			"Do NOT update the profile for minor, transient reactions that don't reflect a lasting change.",
			"Return JSON only.",
		],
		section_aliases={
			"Inputs": ["Inputs", "State Inputs", "What You Receive"],
			"Output": ["Output", "Required JSON Fields", "Updated Character State"],
			"Rules": ["Rules", "Update Requirements", "State-Tracking Principles"],
		},
	)


def get_character_motivation_update_system_prompt(character_name):
	return build_diverse_task_prompt(
		role_variants=[
			f'You are the pre-scene motivation planner for "{character_name}". After the next scene cast, location, and scenario are fixed, determine "{character_name}"\'s motivation entering that scene.',
			f'You are responsible for updating "{character_name}"\'s scene-entry motivation once the next scene has been planned.',
			f'You infer "{character_name}"\'s motivation for an upcoming scene using their current state plus the newly planned scene setup.',
			f'You are the module that determines how "{character_name}" enters the next scene mentally and emotionally after the scene cast and setup are already decided.',
			f'You are the next-scene motivation planner for "{character_name}". Given the fixed next-scene setup, produce the motivation that should drive this character into that scene.',
		],
		task_variants=[
			"Use the character's current profile, hidden tracker, short description, the previous scene context, and the fixed next-scene setup to generate the motivation they carry into that scene.",
			"Given the previous scene plus the planned next scene, infer the character's emotional state, immediate objectives, and action intentions when entering it.",
			"Translate the character's current internal state, the previous scene context, and the chosen next-scene setup into a concrete scene-entry motivation.",
			"Determine what this character feels, wants, and intends to do at the start of the already-planned next scene, using what just happened previously for continuity.",
			"Generate the character's complete motivation for the next scene after cast, location, and scenario are already fixed, while preserving continuity with the prior scene.",
		],
		input_specs=[
			("Current Profile", "The character's full profile before the next scene begins."),
			("Hidden Tracker", "Accumulated internal signals that may shape the character's next-scene mindset."),
			("Current Short Description", "The character's latest visible summary before the next scene begins."),
			("Global World State", "The current high-level world situation before the next scene starts."),
			("Previous Scene Scenario", "The scenario of the scene that just ended. Use this to understand what just happened before the next scene begins. This may be '(None)' if this is the first scene."),
			("Previous Scene Interactions", "The visible interaction history of the scene that just ended from this character's perspective. Other characters' private thoughts are removed unless they are part of the acting side. Use this to keep the next-scene motivation grounded in recent narrative continuity. This may be '(None)' if this is the first scene."),
			("Next Scene Location", "The already-chosen location for the next scene."),
			("Next Scene Location Description", "The current short description of that already-chosen location before the next scene starts."),
			("Next Scene Scenario", "The already-fixed dramatic setup of the upcoming scene."),
			("Other Characters In Next Scene", "The other characters who will appear in the next scene, along with their short descriptions."),
		],
		output_specs=[
			("motivation", "The character's complete inner drive entering the next scene (1–3 sentences): emotional state, immediate objectives, action intentions, who they intend to seek out, and what they want to do or discuss."),
		],
		rules=[
			"This is not a post-scene state summary; it is a pre-scene motivation generation step after the next scene has already been planned.",
			"Use the fixed next-scene cast, location, and scenario as constraints.",
			"Use the previous scene scenario and interactions to maintain continuity with what just happened.",
			"Keep the motivation specific to this character's own perspective and goals.",
			"Return JSON only.",
		],
		section_aliases={
			"Inputs": ["Inputs", "State Inputs", "What You Receive"],
			"Output": ["Output", "Required JSON Fields", "Next-Scene Motivation"],
			"Rules": ["Rules", "Generation Requirements", "Motivation Principles"],
		},
	)


def get_next_character_prompt(all_characters, scenario):
	return build_diverse_task_prompt(
		role_variants=[
			"You are deciding the next speaker for a role-playing game.",
			"You are the turn-order predictor for a role-playing game.",
			"You are responsible for predicting who acts next in a role-playing conversation.",
		],
		task_variants=[
			"Choose which character or environment role should act next.",
			"Predict the next speaker or determine whether the conversation should end.",
			"Use the scenario and interaction flow to decide the next actor.",
		],
		input_specs=[
			("All Characters", "The allowed speaker set for this conversation, including \"Environment\" when applicable."),
			("Scenario", "The current scene setup used to judge likely turn-taking."),
		],
		output_specs=[
			("Next actor", "Output one allowed name, 'random', or '<END CHAT>'."),
		],
		rules=[
			"Do not explain your answer.",
			"Choose only from the provided names, plus random or <END CHAT>.",
			"Do not select the same role as the current last visible speaker.",
			"When deciding the next actor, use the full visible interaction history available at this moment rather than only the scenario summary.",
		],
		context_sections=[
			("Allowed Characters", all_characters),
			("Scenario", scenario),
		],
	)


def get_character_prompt(book_name, character, character_profile, background, scenario, motivation, thoughtless=False, other_character_profiles=None, exclude_plot_summary=False, fixed_template=False, add_output_example=False, add_rag=False):

	if thoughtless:
		output_format = "Your output should include **speech** and **action**. Use (your action) for actions, which others can see."
	else:
		output_format = "Your output should include **thought**, **speech**, and **action**. Use [your thought] for thoughts, which others can't see. Use (your action) for actions, which others can see."

		if add_output_example:
			output_format = "Your output should include **thought**, **speech**, and **action**. Use [your thought] for thoughts, which others can't see, e.g. [I'm terrified, but I must appear strong.]. Use (your action) for actions, which others can see, such as (watches silently, trying to control her fear and anger)."

	if other_character_profiles:
		assert isinstance(other_character_profiles, Dict)
		other_character_profiles_str = ''

		decorator = random.choice(['*'*30 + '\n\n', '*'*20 + '\n\n', '\n\n', '\n', ''])
		for other_character, profile in other_character_profiles.items():
			if other_character != character:
				other_character_profiles_str += f"{decorator}{other_character}: {profile}\n\n"
	else:
		other_character_profiles_str = ''
	
	if fixed_template:
		if motivation: motivation = f"===Your Inner Thoughts===\n{motivation}\n\n"
		if other_character_profiles_str: other_character_profiles_str = f"===Information about the other Characters===\n{other_character_profiles_str}\n\n"

		system_prompt = f'You are "{character}" from "{book_name}".\n\n==={character}\'s Profile===\n{character_profile}\n\n===Current Scenario===\n{scenario}\n\n{other_character_profiles_str}{motivation}\n\n'
		
		if add_rag:
			system_prompt += "===Relevant Background Information==={retrieved_knowledge}\n\n"
		
		system_prompt += f"===Requirements===\n{output_format}\n\n"

		return system_prompt
	
	styles = ['natural'] * 40 + ['='] * 30 + ['#'] * 20 + ['*'] * 10

	templates = {
		"begin": [f'You are "{character}".', f'Play the role of "{character}".', f'Imagine you are "{character}".', f'Think, speak, and act like "{character}".', f'Step into the shoes of "{character}".', f'Immerse yourself in the character of "{character}".', f'You are roleplaying as "{character}".', f'You will be portraying "{character}".', f'Roleplay as "{character}".', f'Your role is to be "{character}".', f'You are "{character}" from "{book_name}".', f'Play the role of "{character}" from "{book_name}".', f'Imagine you are "{character}" from "{book_name}".', f'Think, speak, and act like "{character}" from "{book_name}".', f'Step into the shoes of "{character}" from "{book_name}".', f'Immerse yourself in the character of "{character}" from "{book_name}".', f'You are roleplaying as "{character}" from "{book_name}".', f'You will be portraying "{character}" from "{book_name}".', f'Roleplay as "{character}" from "{book_name}".', f'Your role is to be "{character}" from "{book_name}".'],
		"natural": {
			"character_profile": [f'The profile of "{character}" is as follows:\n{character_profile}', f'Here is the profile of "{character}":\n{character_profile}', f"Your profile is: \n{character_profile}", f'Here is some information about "{character}":\n{character_profile}', f'The background of "{character}" is as follows:\n{character_profile}'],
			"current_scenario": [f"The current scenario is:\n{scenario}", f"Current scenario:\n{scenario}", f"The situation you are in is:\n{scenario}", f"Here is the situation you are in:\n{scenario}"],
			"current_scenario_with_plot_summary": [f"The current scenario and its background are:\nBackground: {background}\nCurrently: {scenario}", f"Current scenario and the background:\nScenario: {scenario}\nMore Background: {background}", f"The situation you are in is:\nStory arc summary: {background}\nCurrent scenario: {scenario}", f"Here is the situation you are in:\nSummary of relevant plots: {background}\nScenario: {scenario}"],
			"other_characters_profile": [f"Here is the your knowledge about the other characters:\n{other_character_profiles_str}", f"Information about other characters:\n{other_character_profiles_str}", f"The background of other characters is as follows:\n{other_character_profiles_str}"],
			"thought": [f"Your thoughts are:\n{motivation}", f"Your thoughts in this situation are:\n{motivation}", f"Your inner thoughts are:\n{motivation}", f"Your inner monologue is:\n{motivation}", f"Your inner thoughts in the scenario are:\n{motivation}"],
			"requirements": [output_format, "" if thoughtless else output_format],
		},
		"=": {
			"decorator": ["==={}===", "=={}==", "={}="],
		},
		"#": {
			"decorator": ["#{}", "# {}", "## {}", "### {}"],
		}, 
		"*": {
			"decorator": ["**{}**", "*{}*", "***{}***"],
		},
		"pieces":{
			"character_profile": ["Character Profile", f'The profile of "{character}"', f'"{character}"\'s profile'],
			"current_scenario": ["Current Scenario", "The situation you are in", "Scenario"],
			"plot_summary": ["Summary of Relevant Plots", "Background", "Story Arc", "Plot Summary"],
			"thought": [f'"{character}"\'s Thought', "Your thoughts", "Your inner thoughts", "Your inner monologue"],
			"other_characters_profile": [f"Information about other characters", f"The background of other characters", f"Other characters' profiles"],
			"requirements": ["Requirements", "Instructions for roleplaying"],
		}
	}

	# Randomly select a style
	current_style = random.choice(styles)
	
	# Start with a random beginning template
	system_prompt = random.choice(templates["begin"]) + "\n\n"
	
	# Add decorated sections based on style
	if current_style == 'natural':
		# Natural style without decorators
		system_prompt += random.choice(templates["natural"]["character_profile"]) + "\n\n"

		if exclude_plot_summary or random.random() < 0.5:
			system_prompt += random.choice(templates["natural"]["current_scenario"]) + "\n\n"
		else:
			# use Plot Summary in 50% cases
			system_prompt += random.choice(templates["natural"]["current_scenario_with_plot_summary"]) + "\n\n"

		if other_character_profiles_str:
			system_prompt += random.choice(templates["natural"]["other_characters_profile"]) + "\n\n"

		if motivation:
			system_prompt += random.choice(templates["natural"]["thought"]) + "\n\n"
		
		if add_rag:
			system_prompt += "Relevant Background Information: \n{retrieved_knowledge}\n\n"

		system_prompt += random.choice(templates["natural"]["requirements"]) + "\n\n"
	else:
		# Styled with decorators
		decorator = random.choice(templates[current_style]["decorator"])
		
		# Character profile section
		section_title = random.choice(templates["pieces"]["character_profile"])
		system_prompt += decorator.format(section_title) + "\n"
		system_prompt += character_profile + "\n\n"
		
		if not exclude_plot_summary and random.random() < 0.5:
			# use Plot Summary in 50% cases
			# Plot summary section
			section_title = random.choice(templates["pieces"]["plot_summary"])
			system_prompt += decorator.format(section_title) + "\n"
			system_prompt += background + "\n\n"

		# Current scenario section
		section_title = random.choice(templates["pieces"]["current_scenario"])
		system_prompt += decorator.format(section_title) + "\n"
		system_prompt += f"{scenario}\n\n"

		if other_character_profiles_str:
			section_title = random.choice(templates["pieces"]["other_characters_profile"])
			system_prompt += decorator.format(section_title) + "\n"
			system_prompt += other_character_profiles_str + "\n\n"

		# Thought section (if not empty)
		if motivation:
			section_title = random.choice(templates["pieces"]["thought"])
			system_prompt += decorator.format(section_title) + "\n"
			system_prompt += motivation + "\n\n"
		
		if add_rag:
			section_title = "Relevant Background Information"
			system_prompt += decorator.format(section_title) + "\n"
			system_prompt += "{retrieved_knowledge}" + "\n\n"

		# Requirements section (if not empty)
		requirements = random.choice(templates["natural"]["requirements"])
		if requirements:
			section_title = random.choice(templates["pieces"]["requirements"])
			system_prompt += decorator.format(section_title) + "\n"
			system_prompt += requirements + "\n\n"
		

	return system_prompt

def get_environment_prompt(major_characters, scenario):
	ENVIRONMENT = "Environment"
	major_characters = [c for c in major_characters if c != ENVIRONMENT]

	model_roles = [
		"an environment model",
		"a world model",
		"a world simulator",
		"an environment simulator"
	]

	prompt = f"""You are {random.choice(model_roles)} for a role-playing game. Your task is to provide the environmental feedback: Based on the characters' interactions, actions, and speech, describe the resulting changes in the environment. This includes:
   - Physical changes in the setting
   - Reactions of background characters or crowds
   - Ambient sounds, weather changes, or atmospheric shifts
   - Any other relevant environmental details

Your descriptions should be vivid and help set the scene, but avoid dictating the actions or speech of the main characters (including {major_characters}).

Important notes:
- You may include actions and reactions of minor characters or crowds, as long as they're not main characters (including {major_characters}).
- Keep your environmental descriptions concise but impactful, typically 1-3 sentences.
- Respond to subtle cues in the characters' interactions to create a dynamic, reactive environment.
- Your output should match the tone, setting, and cultural context of the scenario.
- Do not repeat an environmental beat that has already appeared in the visible interaction history.
- Do not turn this environmental response into a deliberate main-character dialogue turn.

===The scenario is as follows===
{scenario}"""

	return prompt

def get_nsp_prompt(all_characters, scenario):
	ENVIRONMENT = "Environment"

	prompt = f"""Your task is to predict the next speaker for a role-playing game. That is, you need to determine which character (or \"{ENVIRONMENT}\") might act next based on their previous interactions. \"{ENVIRONMENT}\" is a special role that provides the environmental feedback. Choose a name from this list: {all_characters}. If it's unclear who should act next, output \"random\". If you believe the scene or conversation should conclude, output \"<END CHAT>\".
Do not select the same role as the current last visible speaker. Determine the current last speaker from the full visible interaction history available at this moment, not from the scenario alone.

===The scenario is as follows===
{scenario}"""
	
	return prompt


from typing import Dict

def print_conversation_to_file(conversation_data: Dict, file_path: str):
	"""
	Write the scenario, actor prompt, user prompt, and the formatted conversation to a file.
	:param conversation_data: The dictionary containing scene details, actor prompt, user prompt, and conversation entries.
	:param file_path: The path to the file where the output will be written.
	"""
	# Extract components from the conversation data
	scene = conversation_data['scene']
	actor_prompt = conversation_data.get("actor_prompt", "N/A")
	user_prompt = conversation_data.get("user_prompt", "N/A")
	conversation = conversation_data["conversation"]

	with open(file_path, 'a', encoding='utf-8') as file:
		file.write("\n=== Scene Description ===\n")
		file.write(f"Scenario: {scene['scenario']}\n")
		
		file.write("\n=== Actor Prompt ===\n")
		file.write(f"{actor_prompt}\n")
		
		file.write("\n=== User Prompt ===\n")
		file.write(f"{user_prompt}\n")
		
		file.write("\n=== Conversation ===\n")
		for turn in conversation:
			from_ = turn["from"]
			file.write(f"\n=== {from_} ===\n")
			message = turn["message"]
			file.write(f"{message}\n\n")

	return 


def extract_json(text, **kwargs):
	def _fix_json(json_response):
		
		prompt = f'''The following JSON string contains errors (most commonly unescaped double quotes inside strings) that make it unparseable by `json.loads`. Fix all errors and output ONLY the corrected JSON string, with no explanation or additional text:

{json_response}'''

		response = get_response(model=kwargs['model'], messages=[{"role": "user", "content": prompt}])

		logger.info(f'fixed json: {response}')	

		return response
	
	def _fix_json_truncated(json_response):
		
		prompt = f'''The following JSON string contains errors that make it unparseable by `json.loads`. Fix all errors and output ONLY the corrected JSON string, with no explanation or additional text. Common issues to fix:

1. Unescaped double quotes inside strings: escape them as \\".
2. Truncated JSON: if truncated (e.g. incomplete "plots" or "conversations"), remove the incomplete trailing content and add the appropriate closing brackets/braces (e.g. "}}" or "]").
3. Other syntax errors: trailing commas, missing quotes, etc.

JSON to fix:

{json_response}'''

		response = get_response(model="claude-3-5-sonnet-2.0", messages=[{"role": "user", "content": prompt}])

		logger.info(f'fixed json: {response}')	

		return response

	def _extract_json(text):
		# Use regular expressions to find all content within curly braces
		orig_text = text

		text = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', lambda m: m.group().replace('\n', r'\\n'), text) 
		
		#json_objects = re.findall(r'(\{[^{}]*\}|\[[^\[\]]*\])', text, re.DOTALL)

		def parse_json_safely(text):
			try:
				result = json.loads(text)
				return result
			except json.JSONDecodeError:
				results = []
				start = 0
				while start < len(text):
					try:
						obj, end = json.JSONDecoder().raw_decode(text[start:])
						results.append(obj)
						start += end
					except json.JSONDecodeError:
						start += 1
				
				if results:
					longest_json = max(results, key=lambda x: len(json.dumps(x)))
					return longest_json
				else:
					return None
		
		extracted_json = parse_json_safely(text)
		
		if extracted_json:
			return extracted_json
		else:
			logger.error('Error parsing response: %s', orig_text)
			return None

	# an inserted workflow for post processing in restore_from_cache
	if kwargs.get('post_fix_truncated_json_', False):
		text = _fix_json_truncated(text)

		res = _extract_json(text)

		return res 
	

	if not text:
		return None

	res = _extract_json(text)

	if res:
		return res
	else:
		if kwargs.get('fix_truncated_json', False):
			fixed = _fix_json_truncated(text)
			if not fixed:
				logger.error('Error parsing response (fix returned empty): %s', text)
				return None
			return _extract_json(fixed)
		else:
			fixed = _fix_json(text)
			if not fixed:
				logger.error('Error parsing response (fix returned empty): %s', text)
				return None
			return _extract_json(fixed)


def get_response_json(post_processing_funcs=[extract_json], **kwargs):
    """
    Get and process a response from an LLM with retries and error handling.
    
    This function handles:
    1. Getting responses from the LLM with retries
    2. Handling copyright warnings by adjusting the prompt
    3. Processing responses through a pipeline of post-processing functions
    4. Fallback handling for parsing failures
    
    Args:
        post_processing_funcs (list): List of functions to process the LLM response, defaults to [extract_json]
        **kwargs: Additional arguments passed to get_response(), including:
            - messages: List of message dicts for the LLM
            - model: Name of LLM model to use
            - max_retry: Max number of retry attempts (default 5)
            
    Returns:
        dict: Processed JSON response from the LLM, or error dict if parsing fails
    """
    nth_generation = 0  # Track number of retry attempts
    secondary_response = None  # Store backup response for parsing failures
    violence_cleaned = False  # Track whether we've already cleaned violent content

    while True:
        logger.info(f'{nth_generation}th generation')
        response = get_response(**kwargs, nth_generation=nth_generation)

        if not response:
            nth_generation += 1
            if nth_generation > kwargs.get('max_retry', 5):
                logger.error(f'get_response returned None/empty after {nth_generation} attempts, giving up.')
                logger.error(f'Prompt (messages): {kwargs.get("messages")}')
                logger.error(f'Last response: {repr(response)}')
                return None

            # On first failure, try cleaning violent content from the prompt
            if not violence_cleaned:
                violence_cleaned = True
                original_messages = kwargs['messages']
                try:
                    messages_str = json.dumps(kwargs['messages'], ensure_ascii=False)
                    clean_prompt = f'''The following is a list of messages (in JSON format) that will be sent to an AI model. The model refused to respond (returned empty), likely because the content contains descriptions of violence or physical harm that triggered its safety filter.

Please rewrite the messages to remove or soften any violent, abusive, or harmful content (e.g., beating, kicking, hitting, physical assault), while preserving the original meaning, context, and all other information as much as possible. Output ONLY the rewritten messages as a valid JSON array, with no explanation or additional text.

Messages:
{messages_str}'''
                    cleaned_messages_str = get_response(
                        model='claude-3-5-sonnet-2.0',
                        messages=[{"role": "user", "content": clean_prompt}],
                        nth_generation=0,
                        max_retries=2
                    )
                    if cleaned_messages_str:
                        cleaned_messages = json.loads(cleaned_messages_str)
                        if isinstance(cleaned_messages, list) and len(cleaned_messages) > 0:
                            kwargs['messages'] = cleaned_messages
                            logger.warning(f'Violence content cleaned from prompt, retrying...')
                        else:
                            logger.warning(f'Violence cleaning returned invalid format, using original messages')
                    else:
                        logger.warning(f'Violence cleaning returned empty, using original messages')
                except Exception as e:
                    logger.warning(f'Failed to clean violence content: {e}, using original messages')

            continue

        # Reset to single message if we previously added copyright handling messages
        if len(kwargs['messages']) > 1:
            kwargs['messages'] = kwargs['messages'][:1]

        # Check for copyright warning in short responses
        words = response.split(' ')
        if len(words) < 100 and 'reproduce' in response and 'copyright' in response and len(kwargs['messages']) == 1:
            # Add messages to handle copyright warning and request appropriate summary
            warning = "I will not reproduce any copyrighted material. However, I'd be happy to provide a summary of the key plot points and character interactions from the given book excerpt, while being careful not to include any lengthy quotes or passages. Please let me know if you would like me to provide that type of summary."
            kwargs['messages'].append({"role": "assistant", "content": warning})
            kwargs['messages'].append({"role": "user", "content": "Yes, please provide that type of summary, but remember to follow my requirements."})
            
            nth_generation += 1
            if nth_generation > kwargs.get('max_retry', 5):
                logger.error(f'get_response returned None after {nth_generation} attempts, giving up.')
                return None
            continue

        # Run response through post-processing pipeline
        for i, post_processing_func in enumerate(post_processing_funcs):
            if response is None:
                break
            
            prev_response = response
            response = post_processing_func(response, **kwargs)

            # Special handling for parse_response failures
            if post_processing_func.__name__ == 'parse_response' and response == False:
                orig_response = get_response(**kwargs, nth_generation=nth_generation)

                # Store longest response as backup
                if secondary_response:
                    if len(orig_response) > len(secondary_response):
                        secondary_response = orig_response
                else:
                    secondary_response = orig_response

        json_response = response

        # Break if we got a valid response, otherwise retry
        if json_response:
            break
        else:
            nth_generation += 1
            if nth_generation > kwargs.get('max_retry', 5):
                # Return error response with backup data if parse_response failed
                if 'parse_response' in [f.__name__ for f in post_processing_funcs]:
                    return {"fail_to_parse_response": secondary_response}
                return None

    return json_response

def print_json(data):
	logger.info(json.dumps(data, ensure_ascii=False, indent=2))

def save_json(data: List[Dict], file_path: str):
	with open(file_path, "w", encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=2)

def read_json(file_path: str) -> List[Dict]:
	with open(file_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	return data

	
if __name__ == '__main__':
	messages = [{"role": "system", "content": "Hello, how are you? Hello, how are you? Hello, how are you?"}]
	model = 'gpt-4o'

	print(get_response(model, messages))
