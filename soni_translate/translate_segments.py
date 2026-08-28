from tqdm import tqdm
from deep_translator import GoogleTranslator
from itertools import chain
import copy
from .language_configuration import fix_code_language, INVERTED_LANGUAGES
from .logging_setup import logger
import re
import json
import time
import urllib.error
import urllib.request

TRANSLATION_PROCESS_OPTIONS = [
    "google_translator_batch",
    "google_translator",
    "gpt-3.5-turbo-0125_batch",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview_batch",
    "gpt-4-turbo-preview",
    "disable_translation",
]
DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "google_translator",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    "disable_translation",
]


def translate_iterative(segments, target, source=None):
    """
    Translate text segments individually to the specified language.

    Parameters:
    - segments (list): A list of dictionaries with 'text' as a key for
        segment text.
    - target (str): Target language code.
    - source (str, optional): Source language code. Defaults to None.

    Returns:
    - list: Translated text segments in the target language.

    Notes:
    - Translates each segment using Google Translate.

    Example:
    segments = [{'text': 'first segment.'}, {'text': 'second segment.'}]
    translated_segments = translate_iterative(segments, 'es')
    """

    segments_ = copy.deepcopy(segments)

    if (
        not source
    ):
        logger.debug("No source language")
        source = "auto"

    translator = GoogleTranslator(source=source, target=target)

    for line in tqdm(range(len(segments_))):
        text = segments_[line]["text"]
        translated_line = translator.translate(text.strip())
        segments_[line]["text"] = translated_line

    return segments_


def verify_translate(
    segments,
    segments_copy,
    translated_lines,
    target,
    source
):
    """
    Verify integrity and translate segments if lengths match, otherwise
    switch to iterative translation.
    """
    if len(segments) == len(translated_lines):
        for line in range(len(segments_copy)):
            logger.debug(
                f"{segments_copy[line]['text']} >> "
                f"{translated_lines[line].strip()}"
            )
            segments_copy[line]["text"] = translated_lines[
                line].replace("\t", "").replace("\n", "").strip()
        return segments_copy
    else:
        logger.error(
            "The translation failed, switching to google_translate iterative. "
            f"{len(segments), len(translated_lines)}"
        )
        return translate_iterative(segments, target, source)


def translate_batch(segments, target, chunk_size=2000, source=None):
    """
    Translate a batch of text segments into the specified language in chunks,
        respecting the character limit.

    Parameters:
    - segments (list): List of dictionaries with 'text' as a key for segment
        text.
    - target (str): Target language code.
    - chunk_size (int, optional): Maximum character limit for each translation
        chunk (default is 2000; max 5000).
    - source (str, optional): Source language code. Defaults to None.

    Returns:
    - list: Translated text segments in the target language.

    Notes:
    - Splits input segments into chunks respecting the character limit for
        translation.
    - Translates the chunks using Google Translate.
    - If chunked translation fails, switches to iterative translation using
        `translate_iterative()`.

    Example:
    segments = [{'text': 'first segment.'}, {'text': 'second segment.'}]
    translated = translate_batch(segments, 'es', chunk_size=4000, source='en')
    """

    segments_copy = copy.deepcopy(segments)

    if (
        not source
    ):
        logger.debug("No source language")
        source = "auto"

    # Get text
    text_lines = []
    for line in range(len(segments_copy)):
        text = segments_copy[line]["text"].strip()
        text_lines.append(text)

    # chunk limit
    text_merge = []
    actual_chunk = ""
    global_text_list = []
    actual_text_list = []
    for one_line in text_lines:
        one_line = " " if not one_line else one_line
        if (len(actual_chunk) + len(one_line)) <= chunk_size:
            if actual_chunk:
                actual_chunk += " ||||| "
            actual_chunk += one_line
            actual_text_list.append(one_line)
        else:
            text_merge.append(actual_chunk)
            actual_chunk = one_line
            global_text_list.append(actual_text_list)
            actual_text_list = [one_line]
    if actual_chunk:
        text_merge.append(actual_chunk)
        global_text_list.append(actual_text_list)

    # translate chunks
    progress_bar = tqdm(total=len(segments), desc="Translating")
    translator = GoogleTranslator(source=source, target=target)
    split_list = []
    try:
        for text, text_iterable in zip(text_merge, global_text_list):
            translated_line = translator.translate(text.strip())
            split_text = translated_line.split("|||||")
            if len(split_text) == len(text_iterable):
                progress_bar.update(len(split_text))
            else:
                logger.debug(
                    "Chunk fixing iteratively. Len chunk: "
                    f"{len(split_text)}, expected: {len(text_iterable)}"
                )
                split_text = []
                for txt_iter in text_iterable:
                    translated_txt = translator.translate(txt_iter.strip())
                    split_text.append(translated_txt)
                    progress_bar.update(1)
            split_list.append(split_text)
        progress_bar.close()
    except Exception as error:
        progress_bar.close()
        logger.error(str(error))
        logger.warning(
            "The translation in chunks failed, switching to iterative."
            " Related: too many request"
        )  # use proxy or less chunk size
        return translate_iterative(segments, target, source)

    # un chunk
    translated_lines = list(chain.from_iterable(split_list))

    return verify_translate(
        segments, segments_copy, translated_lines, target, source
    )

_TRANSLATE_PROMPT = (
    "You are an Italian dubbing adapter for an irreverent adult animated comedy.\n"
    "Adapt the English line into natural, colloquial SPOKEN Italian, the way it "
    "would actually be dubbed — not a literal or formal translation.\n"
    "- Register: italiano parlato, informale. Always \"tu\", contractions, real interjections.\n"
    "- Prioritize rhythm, punch, and comedic timing over word-for-word fidelity.\n"
    "- Localize idioms, slang and cultural references to Italian equivalents.\n"
    "- Keep crude language and profanity when the original has it — do NOT soften or censor.\n"
    "- Keep it short and speakable (it has to fit the character's mouth).\n"
    "Output ONLY the adapted Italian line — no notes, no alternatives, no quotes.\n"
    "\n"
    "Examples of the target style:\n"
    "- \"Are you kidding me right now?\" -> \"Ma mi stai prendendo per il culo?\"\n"
    "- \"That's the dumbest thing I've ever heard.\" -> \"È la cazzata più grossa che abbia mai sentito.\"\n"
    "- \"Whatever, I don't even care.\" -> \"Vabbè, non me ne frega un cazzo.\"\n"
    "- \"Dude, this is so messed up.\" -> \"Raga, è tutto una merda.\"\n"
    "- \"Wow, that's cool!.\" -> \"Wow fico!\"\n"
    "\n"
    "English line:\n"
    "{TEXT}"
)

def call_ollama_generate(
    model,
    original_text 
):
    url = "http://192.168.178.210:11434/api/generate"
    prompt = _TRANSLATE_PROMPT.replace("{TEXT}", original_text)
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 512},
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )

    last_error = None
    retries = 4
    for attempt in range(1, retries): #retries before giving up
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
        except (urllib.error.URLError, TimeoutError,
                json.JSONDecodeError) as error:
            last_error = error
            if attempt == 4:
                break
            delay = _OLLAMA_BACKOFF * (2 ** (attempt - 1))
            logger.warning(
                f"/api/generate attempt {attempt}/{retries} "
                f"failed ({error}); retrying in {delay:.1f}s"
            )
            time.sleep(delay)
    raise RuntimeError(
        f"/api/generate failed after {retries} attempts: "
        f"{last_error}"
    )

def call_gpt_translate(
    client,
    model,
    system_prompt,
    user_prompt,
    original_text=None,
    batch_lines=None,
):

    # https://platform.openai.com/docs/guides/text-generation/json-mode
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
        ]
    )
    result = response.choices[0].message.content
    logger.debug(f"Result: {str(result)}")

    try:
        #translation = json.loads(result)
        #translation = result.strip()
        translation = json.loads(result).get('choices', [{}])[0].get('message', {}).get('content')
        # tolerate an empty JSON object {} — fall back to the source text for this segment
        if isinstance(translation, dict) and not translation:
            translation = user_prompt
    except Exception as error:
        match_result = re.search(r'\{.*?\}', result)
        if match_result:
            logger.error(str(error))
            json_str = match_result.group(0)
            translation = json.loads(json_str)
            if isinstance(translation, dict) and not translation:
                translation = user_prompt
        else:
            # nothing parseable at all — keep the source text, don't crash the run
            logger.error(f"Unparseable translation, keeping source: {str(error)}")
            translation = user_prompt

    # Get valid data
    if batch_lines:
        for conversation in translation.values():
            if isinstance(conversation, dict):
                conversation = list(conversation.values())[0]
            if (
                list(
                    original_text["conversation"][0].values()
                )[0].strip() ==
                list(conversation[0].values())[0].strip()
            ):
                continue
            if len(conversation) == batch_lines:
                break

        fix_conversation_length = []
        for line in conversation:
            for speaker_code, text_tr in line.items():
                fix_conversation_length.append({speaker_code: text_tr})

        logger.debug(f"Data batch: {str(fix_conversation_length)}")
        logger.debug(
            f"Lines Received: {len(fix_conversation_length)},"
            f" expected: {batch_lines}"
        )

        return fix_conversation_length

    else:
        # Unwrap whatever shape the model returned (dict / nested list / set / str)
        # down to the first string; never crash the run on an odd shape.
        def _first_str(x, _depth=0):
            if _depth > 10:
                return ""
            if isinstance(x, str):
                return x
            if isinstance(x, dict):
                vals = list(x.values())
                return _first_str(vals[0], _depth + 1) if vals else ""
            if isinstance(x, (list, tuple, set)):
                items = list(x)
                return _first_str(items[0], _depth + 1) if items else ""
            return str(x) if x is not None else ""

        translation = _first_str(translation) or user_prompt

        return translation

def gpt_sequential(segments, model, target, source=None):
    from openai import OpenAI

    translated_segments = copy.deepcopy(segments)

    client = OpenAI()
    progress_bar = tqdm(total=len(segments), desc="Translating")

    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[target]).strip()
    lang_sc = ""
    if source:
        lang_sc = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[source]).strip()

    fixed_target = fix_code_language(target)
    fixed_source = fix_code_language(source) if source else "auto"

    #system_prompt = "Machine translation designed to output the translated_text JSON."
    #system_prompt = "You are a professional English (en) to Italian (it) translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Italian grammar, vocabulary, and cultural sensitivities.\nProduce only the Italian translation, without any additional explanations or commentary. Please translate the following English text into Italian:\n\n\n"
    system_prompt = "You are a professional English (en) to Italian (it) translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Italian grammar, vocabulary, and cultural sensitivities. Produce only on single Italian translation, without any additional explanations, commentary or alternative translations."

    for i, line in enumerate(translated_segments):
        text = line["text"].strip()
        start = line["start"]
        #user_prompt = f"Translate the following {lang_sc} text into {lang_tg}, write the fully translated text and nothing more:\n{text}"
        #user_prompt = f"You are a professional English (en) to Italian (it) translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Italian grammar, vocabulary, and cultural sensitivities. Produce only the Italian translation, without any additional explanations or commentary.\n\nPlease translate the following English text into Italian:\n\n\n{text}"
        #user_prompt = f"{text}"
        #user_prompt = (
        #    "You are a professional English (en) to Italian (it) translator. "
        #    "Your goal is to accurately convey the meaning and nuances of the original English text "
        #    "while adhering to Italian grammar, vocabulary, and cultural sensitivities. "
        #    "Produce only the Italian translation, without any additional explanations or commentary. "
        #    "Please translate the following English text into Italian:\n\n\n"
        #    f"{text}"
        #)
        user_prompt = f"Please translate the following English text into Italian:\n\n\n{text}"

        time.sleep(0.5)

        try:
            #translated_text = call_gpt_translate(
            #    client,
            #    model,
            #    system_prompt,
            #    user_prompt,
            #)
            translated_text = call_ollama_generate(
                model,
                text
            )
            logger.debug(f"{translated_text}")

        except Exception as error:
            logger.error(
                f"{str(error)} >> The text of segment {start} "
                "is being corrected with Google Translate"
            )
            translator = GoogleTranslator(
                source=fixed_source, target=fixed_target
            )
            translated_text = translator.translate(text.strip())

        translated_segments[i]["text"] = translated_text.strip()
        progress_bar.update(1)

    progress_bar.close()

    return translated_segments


def gpt_batch(segments, model, target, token_batch_limit=900, source=None):
    from openai import OpenAI
    import tiktoken

    token_batch_limit = max(100, (token_batch_limit - 40) // 2)
    progress_bar = tqdm(total=len(segments), desc="Translating")
    segments_copy = copy.deepcopy(segments)
    encoding = tiktoken.get_encoding("cl100k_base")
    client = OpenAI()

    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[target]).strip()
    lang_sc = ""
    if source:
        lang_sc = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[source]).strip()

    fixed_target = fix_code_language(target)
    fixed_source = fix_code_language(source) if source else "auto"

    name_speaker = "ABCDEFGHIJKL"

    translated_lines = []
    text_data_dict = []
    num_tokens = 0
    count_sk = {char: 0 for char in "ABCDEFGHIJKL"}

    for i, line in enumerate(segments_copy):
        text = line["text"]
        speaker = line["speaker"]
        last_start = line["start"]
        # text_data_dict.append({str(int(speaker[-1])+1): text})
        index_sk = int(speaker[-2:])
        character_sk = name_speaker[index_sk]
        count_sk[character_sk] += 1
        code_sk = character_sk+str(count_sk[character_sk])
        text_data_dict.append({code_sk: text})
        num_tokens += len(encoding.encode(text)) + 7
        if num_tokens >= token_batch_limit or i == len(segments_copy)-1:
            try:
                batch_lines = len(text_data_dict)
                batch_conversation = {"conversation": copy.deepcopy(text_data_dict)}
                # Reset vars
                num_tokens = 0
                text_data_dict = []
                count_sk = {char: 0 for char in "ABCDEFGHIJKL"}
                # Process translation
                # https://arxiv.org/pdf/2309.03409.pdf
                system_prompt = f"Machine translation designed to output the translated_conversation key JSON containing a list of {batch_lines} items."
                user_prompt = f"Translate each of the following text values in conversation{' from' if lang_sc else ''} {lang_sc} to {lang_tg}:\n{batch_conversation}"
                logger.debug(f"Prompt: {str(user_prompt)}")

                conversation = call_gpt_translate(
                    client,
                    model,
                    system_prompt,
                    user_prompt,
                    original_text=batch_conversation,
                    batch_lines=batch_lines,
                )

                if len(conversation) < batch_lines:
                    raise ValueError(
                        "Incomplete result received. Batch lines: "
                        f"{len(conversation)}, expected: {batch_lines}"
                    )

                for i, translated_text in enumerate(conversation):
                    if i+1 > batch_lines:
                        break
                    translated_lines.append(list(translated_text.values())[0])

                progress_bar.update(batch_lines)

            except Exception as error:
                logger.error(str(error))

                first_start = segments_copy[max(0, i-(batch_lines-1))]["start"]
                logger.warning(
                    f"The batch from {first_start} to {last_start} "
                    "failed, is being corrected with Google Translate"
                )

                translator = GoogleTranslator(
                    source=fixed_source,
                    target=fixed_target
                )

                for txt_source in batch_conversation["conversation"]:
                    translated_txt = translator.translate(
                        list(txt_source.values())[0].strip()
                    )
                    translated_lines.append(translated_txt.strip())
                    progress_bar.update(1)

    progress_bar.close()

    return verify_translate(
        segments, segments_copy, translated_lines, fixed_target, fixed_source
    )


def translate_text(
    segments,
    target,
    translation_process="google_translator_batch",
    chunk_size=4500,
    source=None,
    token_batch_limit=1000,
):
    """Translates text segments using a specified process."""
    match translation_process:
        case "google_translator_batch":
            return translate_batch(
                segments,
                fix_code_language(target),
                chunk_size,
                fix_code_language(source)
            )
        case "google_translator":
            return translate_iterative(
                segments,
                fix_code_language(target),
                fix_code_language(source)
            )
        case model if model in ["gpt-3.5-turbo-0125", "gpt-4-turbo-preview"]:
            return gpt_sequential(segments, model, target, source)
        case model if model in ["gpt-3.5-turbo-0125_batch", "gpt-4-turbo-preview_batch",]:
            return gpt_batch(
                segments,
                translation_process.replace("_batch", ""),
                target,
                token_batch_limit,
                source
            )
        case "disable_translation":
            return segments
        case _:
            raise ValueError("No valid translation process")
