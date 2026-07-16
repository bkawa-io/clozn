"""No-model tests for clozn.runs.sections (prompt-section influence -- the ablatable-section manifest).

Mirrors test_runlog.py / test_runtime_architecture.py's style: unittest.TestCase, the store isolated by
pointing runlog.RUNS_DIR at a temp dir, a bare-bones handler/substrate stub for exercising the real route
code (no HTTP socket, no model, no GPU, no network -- see clozn.runs.sections's own "stdlib-only, never
raises" contract, which every test here is ultimately checking against).

Layout:
  * SlugAndIdTests / DedupeIdsTests        -- the "sec_" + slug(name) id rule and cross-list uniqueness.
  * SectionsFromMessagesTests              -- explicit `clozn_section` tags (chat shape).
  * AutoChunkMessagesTests                 -- the deterministic auto-chunker, rules a-f.
  * AutoChunkPromptTests                   -- its raw-prompt counterpart (message_index always None).
  * SectionsFromNativeTests                -- explicit char ranges (native shape) + clamping.
  * MemoryCardSectionsTests                -- locating applied memory cards inside the assembled prompt.
  * ResolveTests                           -- reconstructing a section's text from a run dict.
  * StoreRoundTripTests                    -- record() persists `sections`; absent when empty.
  * OpenAIRouteWiringTests                 -- /v1/chat/completions: explicit-wins, auto fallback, the
                                             engine never sees `clozn_section`, backward compat.
  * SSEWiringTests                         -- the streaming path threads `sections` to _log_run too.
  * LogRunMemoryCardWiringTests            -- app.py's _log_run folds applied memory cards into whatever
                                             manifest a route passed in (deliverable 5, exercised through
                                             the real shared helper, not just the standalone function).
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

from clozn.runs import sections as clozn_sections
import clozn.runs.store as runlog
from clozn.server import app
from clozn.server import sse
from clozn.server.routes import openai as openai_routes


# =========================================================================================== fixtures ===

def _pad(sentence: str, times: int) -> str:
    """A body of `sentence` repeated `times` -- comfortably over MIN_CHUNK_CHARS (200) so it survives
    merge-small on its own, used throughout to build >600-char messages with clearly-separated pieces."""
    return (sentence + " ") * times


HEADER_DOC = (
    "# Section One\n" + _pad("alpha content sentence.", 14) +
    "\n\n# Section Two\n" + _pad("beta content sentence.", 14) +
    "\n\n# Section Three\n" + _pad("gamma content sentence.", 14)
)

HR_DOC = (
    _pad("first block sentence.", 14) +
    "\n\n---\n\n" + _pad("second block sentence.", 14)
)

DOC_MARKER_DOC = (
    "Document 1\n" + _pad("first doc content.", 14) +
    "\n\nDocument 2\n" + _pad("second doc content.", 14) +
    "\n\nDocument 3\n" + _pad("third doc content.", 14)
)

PARAGRAPH_DOC = (
    _pad("Paragraph one content.", 12) + "\n\n" +
    _pad("Paragraph two content.", 12) + "\n\n" +
    _pad("Paragraph three content.", 12)
)

_CODE_BODY = "# not a real header\n" + "\n".join(f"value_{i} = {i}" for i in range(20))
FENCED_DOC = (
    _pad("Intro text.", 20) +
    "\n\n```python\n" + _CODE_BODY + "\n```\n\n" +
    _pad("Outro text.", 20)
)

STRONG_HEADER_SURVIVES_DOC = "# Big Section\n" + _pad("filler content.", 40) + "\n\n# Tiny\nshort tail."

# The exact live-run repro from the bug report: a ~176-char chat system message with three "## " headers
# that used to collapse into ONE whole-message auto section (the LONG_MESSAGE_CHARS gate hid the author's
# own structure below 600 chars) -- THE acceptance fixture for the two-tier chunking fix.
REPRO_DOC = (
    "## Style\nAnswer in one short sentence.\n\n"
    "## Reference\nThe capital of the fictional country Zolara is Marrowport, founded in 1847.\n\n"
    "## Example\nQ: What is 2+2? A: The answer is 4."
)

HR_SHORT_DOC = (
    "This is the first half of a short system message right here." +
    "\n\n---\n\n" +
    "This is the second half of that same short system message."
)

NO_MARKER_SHORT_DOC = (
    "This is a short plain message with no markdown headers, no horizontal rules, and no blank-line "
    "paragraph breaks anywhere -- just a couple of ordinary sentences."
)

# A long (>600 char), fully unstructured blob (no headers/hr/doc-markers at all) whose middle paragraph is
# tiny -- exercises the WEAK-tier fallback's own merge-small step (`_merge_small_spans`), which still
# applies to paragraph fragments even though strong header chunks no longer get merged away.
WEAK_MERGE_DOC = (
    _pad("This is a reasonably long filler paragraph about nothing in particular.", 8) +
    "\n\ntiny.\n\n" +
    _pad("Another reasonably long filler paragraph about something else entirely.", 8)
)

# Short (<600 chars) message mixing one REAL header with a fenced block whose content itself starts with
# "#" -- proves protected-span priority holds even when a strong boundary triggers a split at short length.
FENCE_HASH_SHORT = (
    "## Real Header\nintro text before the fence.\n\n"
    "```\n# not a real header, just a comment\nvalue = 1\n```\n\n"
    "more text after the fence."
)


# ======================================================================================================
class SlugAndIdTests(unittest.TestCase):
    def test_slugify_lowercases_and_collapses_punctuation(self):
        self.assertEqual(clozn_sections._slugify("RAG Context!!"), "rag_context")
        self.assertEqual(clozn_sections._slugify("  --weird--  "), "weird")
        self.assertEqual(clozn_sections._slugify(""), "section")

    def test_make_id_suffixes_on_collision(self):
        used: set = set()
        first = clozn_sections._make_id("policy", used)
        second = clozn_sections._make_id("policy", used)
        third = clozn_sections._make_id("policy", used)
        self.assertEqual(first, "sec_policy")
        self.assertEqual(second, "sec_policy_2")
        self.assertEqual(third, "sec_policy_3")


class DedupeIdsTests(unittest.TestCase):
    def test_cross_list_collisions_get_suffixed(self):
        a = [{"id": "sec_card_1", "name": "card_1"}]
        b = [{"id": "sec_card_1", "name": "clozn_section_named_that_too"}]
        combined = clozn_sections.dedupe_ids(a + b)
        ids = [s["id"] for s in combined]
        self.assertEqual(ids, ["sec_card_1", "sec_card_1_2"])
        # the input dicts are never mutated in place
        self.assertEqual(b[0]["id"], "sec_card_1")

    def test_no_collision_is_a_no_op(self):
        a = [{"id": "sec_a", "name": "a"}, {"id": "sec_b", "name": "b"}]
        self.assertEqual(clozn_sections.dedupe_ids(a), a)


# ======================================================================================================
class SectionsFromMessagesTests(unittest.TestCase):
    def test_no_tags_returns_none(self):
        messages = [{"role": "system", "content": "hello"}, {"role": "user", "content": "hi"}]
        self.assertIsNone(clozn_sections.sections_from_messages(messages))

    def test_single_tagged_message_becomes_one_whole_message_section(self):
        messages = [
            {"role": "system", "content": "RAG BLOCK TEXT", "clozn_section": "rag_context"},
            {"role": "user", "content": "the actual question"},
        ]
        out = clozn_sections.sections_from_messages(messages)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 1)
        sec = out[0]
        self.assertEqual(sec["name"], "rag_context")
        self.assertEqual(sec["id"], "sec_rag_context")
        self.assertEqual(sec["source"], "api")
        self.assertEqual(sec["parts"], [{"message_index": 0, "start": 0, "end": len("RAG BLOCK TEXT")}])
        self.assertEqual(sec["char_count"], len("RAG BLOCK TEXT"))
        self.assertEqual(sec["preview"], "RAG BLOCK TEXT"[:80])

    def test_multiple_messages_sharing_a_name_group_into_one_section_with_multiple_parts(self):
        messages = [
            {"role": "system", "content": "policy part one", "clozn_section": "policy"},
            {"role": "assistant", "content": "policy part two", "clozn_section": "policy"},
            {"role": "user", "content": "question"},
        ]
        out = clozn_sections.sections_from_messages(messages)
        self.assertEqual(len(out), 1)
        sec = out[0]
        self.assertEqual(sec["name"], "policy")
        self.assertEqual(len(sec["parts"]), 2)
        self.assertEqual(sec["parts"][0]["message_index"], 0)
        self.assertEqual(sec["parts"][1]["message_index"], 1)
        self.assertEqual(sec["char_count"], len("policy part one") + len("policy part two"))

    def test_final_user_message_is_never_a_section_even_if_tagged(self):
        messages = [
            {"role": "system", "content": "context", "clozn_section": "ctx"},
            {"role": "user", "content": "final question", "clozn_section": "sneaky"},
        ]
        out = clozn_sections.sections_from_messages(messages)
        names = [s["name"] for s in out]
        self.assertEqual(names, ["ctx"])   # "sneaky" (the final turn) never appears

    def test_only_the_final_message_tagged_returns_none(self):
        messages = [
            {"role": "system", "content": "untagged context"},
            {"role": "user", "content": "final question", "clozn_section": "sneaky"},
        ]
        self.assertIsNone(clozn_sections.sections_from_messages(messages))

    def test_non_string_tag_is_ignored(self):
        messages = [
            {"role": "system", "content": "hi", "clozn_section": None},
            {"role": "system", "content": "there", "clozn_section": 3},
            {"role": "user", "content": "q"},
        ]
        self.assertIsNone(clozn_sections.sections_from_messages(messages))

    def test_ids_are_unique_across_distinct_names(self):
        messages = [
            {"role": "system", "content": "a", "clozn_section": "Same Name!"},
            {"role": "assistant", "content": "b", "clozn_section": "same_name"},
            {"role": "user", "content": "q"},
        ]
        out = clozn_sections.sections_from_messages(messages)
        ids = [s["id"] for s in out]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, ["sec_same_name", "sec_same_name_2"])


# ======================================================================================================
class AutoChunkMessagesTests(unittest.TestCase):
    def _messages(self, content):
        return [{"role": "system", "content": content}, {"role": "user", "content": "final question"}]

    def test_determinism_same_input_twice_is_byte_identical(self):
        messages = self._messages(HEADER_DOC)
        first = clozn_sections.auto_chunk_messages(messages)
        second = clozn_sections.auto_chunk_messages(self._messages(HEADER_DOC))
        self.assertEqual(first, second)

    def test_short_message_is_one_whole_chunk_even_with_a_header_inside(self):
        short = "# Title\nshort body under the threshold"
        self.assertLess(len(short), clozn_sections.LONG_MESSAGE_CHARS)
        out = clozn_sections.auto_chunk_messages(self._messages(short))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["parts"], [{"message_index": 0, "start": 0, "end": len(short)}])

    def test_only_the_final_user_message_yields_nothing(self):
        messages = [{"role": "user", "content": "the only message, and it's final"}]
        self.assertEqual(clozn_sections.auto_chunk_messages(messages), [])

    def test_earlier_user_turns_are_never_auto_chunked(self):
        messages = [
            {"role": "user", "content": "an earlier user turn " * 40},   # non-final, but role==user
            {"role": "assistant", "content": "a reply"},
            {"role": "user", "content": "final question"},
        ]
        out = clozn_sections.auto_chunk_messages(messages)
        indices = {p["message_index"] for sec in out for p in sec["parts"]}
        self.assertNotIn(0, indices)   # the earlier USER turn never becomes a section

    def test_markdown_headers_split_into_separate_sections(self):
        out = clozn_sections.auto_chunk_messages(self._messages(HEADER_DOC))
        self.assertEqual(len(out), 3)
        self.assertEqual([s["name"] for s in out], ["auto_1", "auto_2", "auto_3"])
        self.assertTrue(out[0]["preview"].startswith("# Section One"))
        self.assertTrue(out[1]["preview"].startswith("# Section Two"))
        self.assertTrue(out[2]["preview"].startswith("# Section Three"))
        for s in out:
            self.assertEqual(s["source"], "auto")
            self.assertTrue(s["id"].startswith("sec_auto_"))

    def test_horizontal_rule_splits_into_two_sections(self):
        out = clozn_sections.auto_chunk_messages(self._messages(HR_DOC))
        self.assertEqual(len(out), 2)
        self.assertIn("first block", out[0]["preview"])
        self.assertIn("second block", out[1]["preview"])

    def test_document_markers_split_into_separate_sections(self):
        out = clozn_sections.auto_chunk_messages(self._messages(DOC_MARKER_DOC))
        self.assertEqual(len(out), 3)
        self.assertTrue(out[0]["preview"].startswith("Document 1"))
        self.assertTrue(out[1]["preview"].startswith("Document 2"))
        self.assertTrue(out[2]["preview"].startswith("Document 3"))

    def test_paragraph_breaks_are_the_fallback_split(self):
        out = clozn_sections.auto_chunk_messages(self._messages(PARAGRAPH_DOC))
        self.assertEqual(len(out), 3)
        self.assertTrue(out[0]["preview"].startswith("Paragraph one"))
        self.assertTrue(out[1]["preview"].startswith("Paragraph two"))
        self.assertTrue(out[2]["preview"].startswith("Paragraph three"))

    def test_fenced_code_block_kept_intact_as_one_chunk(self):
        out = clozn_sections.auto_chunk_messages(self._messages(FENCED_DOC))
        self.assertEqual(len(out), 3)
        code_section = out[1]
        self.assertIn("```python", code_section["preview"])
        full_text = clozn_sections.resolve(
            {"messages": self._messages(FENCED_DOC), "sections": out}, code_section["id"]
        )
        self.assertIn("# not a real header", full_text)   # the '#' line INSIDE the fence was not split on
        self.assertIn("value_19 = 19", full_text)
        self.assertTrue(full_text.strip().startswith("```python"))
        self.assertTrue(full_text.strip().endswith("```"))

    def test_xml_wrapper_tag_kept_intact_as_one_chunk(self):
        body = (_pad("before the wrapper.", 15) +
                "\n\n<context>\n" + _pad("wrapped content line.", 15) + "\n</context>\n\n" +
                _pad("after the wrapper.", 15))
        out = clozn_sections.auto_chunk_messages(self._messages(body))
        self.assertEqual(len(out), 3)
        middle = clozn_sections.resolve({"messages": self._messages(body), "sections": out}, out[1]["id"])
        self.assertTrue(middle.strip().startswith("<context>"))
        self.assertTrue(middle.strip().endswith("</context>"))

    def test_strong_header_chunk_is_never_merged_even_when_tiny(self):
        """The two-tier policy's core rule: a STRONG (author-marked) boundary's chunk is never folded away
        by merge-small, no matter how small -- unlike the old single-tier chunker, which would have folded
        the two-line "# Tiny" section into "# Big Section" just because it was under MIN_CHUNK_CHARS."""
        self.assertGreater(len(STRONG_HEADER_SURVIVES_DOC), clozn_sections.LONG_MESSAGE_CHARS)
        out = clozn_sections.auto_chunk_messages(self._messages(STRONG_HEADER_SURVIVES_DOC))
        self.assertEqual(len(out), 2)   # "# Tiny" survives as its OWN section, not folded into Big Section
        self.assertTrue(out[0]["preview"].startswith("# Big Section"))
        self.assertTrue(out[1]["preview"].startswith("# Tiny"))
        self.assertLess(out[1]["char_count"], clozn_sections.MIN_CHUNK_CHARS)   # small on purpose
        full_text = clozn_sections.resolve(
            {"messages": self._messages(STRONG_HEADER_SURVIVES_DOC), "sections": out}, out[1]["id"]
        )
        self.assertEqual(full_text, "# Tiny\nshort tail.")

    def test_weak_path_still_merges_a_tiny_paragraph_fragment_in_a_long_unstructured_blob(self):
        """Unlike a strong (header) chunk, a WEAK (paragraph) fragment produced by splitting one long
        unstructured segment is still subject to merge-small -- the fix narrows WHERE merge-small applies,
        it doesn't remove it."""
        self.assertGreater(len(WEAK_MERGE_DOC), clozn_sections.LONG_MESSAGE_CHARS)
        out = clozn_sections.auto_chunk_messages(self._messages(WEAK_MERGE_DOC))
        self.assertEqual(len(out), 2)   # the tiny "tiny." paragraph folded into its preceding neighbor
        first_text = clozn_sections.resolve(
            {"messages": self._messages(WEAK_MERGE_DOC), "sections": out}, out[0]["id"]
        )
        second_text = clozn_sections.resolve(
            {"messages": self._messages(WEAK_MERGE_DOC), "sections": out}, out[1]["id"]
        )
        self.assertIn("tiny.", first_text)
        self.assertIn("nothing in particular", first_text)
        self.assertIn("something else entirely", second_text)

    def test_short_message_with_a_single_hr_splits_into_two_sections(self):
        """A strong boundary (the horizontal rule) splits even a short (<LONG_MESSAGE_CHARS) message --
        the old chunker only looked for structure once a message was already long."""
        self.assertLess(len(HR_SHORT_DOC), clozn_sections.LONG_MESSAGE_CHARS)
        out = clozn_sections.auto_chunk_messages(self._messages(HR_SHORT_DOC))
        self.assertEqual(len(out), 2)
        self.assertIn("first half", out[0]["preview"])
        second_text = clozn_sections.resolve(
            {"messages": self._messages(HR_SHORT_DOC), "sections": out}, out[1]["id"]
        )
        self.assertIn("second half", second_text)

    def test_short_message_with_no_structure_at_all_is_still_one_section(self):
        """No strong markers, no blank-line paragraphs, under LONG_MESSAGE_CHARS -> an honest "nothing to
        split on" single chunk -- the two-tier policy doesn't invent structure that isn't there."""
        self.assertLess(len(NO_MARKER_SHORT_DOC), clozn_sections.LONG_MESSAGE_CHARS)
        out = clozn_sections.auto_chunk_messages(self._messages(NO_MARKER_SHORT_DOC))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["parts"],
                         [{"message_index": 0, "start": 0, "end": len(NO_MARKER_SHORT_DOC)}])

    def test_fenced_hash_inside_a_short_message_is_not_treated_as_a_header(self):
        """A '#' at the start of a line INSIDE a fenced code block is not a markdown header -- the
        protected span wins over the header regex even though the fence's own edges are themselves strong
        boundaries that split this short message into three pieces (not four)."""
        self.assertLess(len(FENCE_HASH_SHORT), clozn_sections.LONG_MESSAGE_CHARS)
        out = clozn_sections.auto_chunk_messages(self._messages(FENCE_HASH_SHORT))
        self.assertEqual(len(out), 3)   # real header, fenced block, trailing text -- NOT split 4 ways
        code_text = clozn_sections.resolve(
            {"messages": self._messages(FENCE_HASH_SHORT), "sections": out}, out[1]["id"]
        )
        self.assertTrue(code_text.strip().startswith("```"))
        self.assertIn("# not a real header", code_text)

    def test_acceptance_three_short_markdown_headers_isolate_each_section(self):
        """THE acceptance test for the two-tier chunking fix. A live end-to-end test proved a single
        ~176-char chat system message with three '## ' headers used to get chunked into ONE section
        covering the whole message, so per-section influence was a useless 100% on the blob and the
        decisive "Reference" fact (Marrowport) was never isolated. Now each '## ' header is a STRONG
        boundary that splits at ANY length, so this short message (well under LONG_MESSAGE_CHARS) still
        yields three sections, one per header -- and the Marrowport fact lands in its OWN section, not
        merged with Style or Example."""
        self.assertLess(len(REPRO_DOC), clozn_sections.LONG_MESSAGE_CHARS)
        out = clozn_sections.auto_chunk_messages(self._messages(REPRO_DOC))
        self.assertEqual(len(out), 3)
        self.assertEqual([s["name"] for s in out], ["auto_1", "auto_2", "auto_3"])
        style, reference, example = out
        self.assertTrue(style["preview"].startswith("## Style"))
        self.assertTrue(reference["preview"].startswith("## Reference"))
        self.assertTrue(example["preview"].startswith("## Example"))
        run = {"messages": self._messages(REPRO_DOC), "sections": out}
        style_text = clozn_sections.resolve(run, style["id"])
        reference_text = clozn_sections.resolve(run, reference["id"])
        example_text = clozn_sections.resolve(run, example["id"])
        self.assertIn("Marrowport", reference_text)
        self.assertNotIn("Marrowport", style_text)     # the decisive fact is isolated, not blended in
        self.assertNotIn("Marrowport", example_text)
        self.assertIn("one short sentence", style_text)
        self.assertIn("2+2", example_text)

    def test_acceptance_repro_is_byte_identical_across_two_runs(self):
        first = clozn_sections.auto_chunk_messages(self._messages(REPRO_DOC))
        second = clozn_sections.auto_chunk_messages(self._messages(REPRO_DOC))
        self.assertEqual(first, second)

    def test_cap_at_16_merges_smallest_adjacent_pairs(self):
        pieces = [f"# Header {i}\n" + _pad(f"content body number {i}.", 12) for i in range(20)]
        text = "\n\n".join(pieces)
        out = clozn_sections.auto_chunk_messages(self._messages(text))
        self.assertLessEqual(len(out), clozn_sections.MAX_SECTIONS)
        self.assertEqual(len(out), 16)
        self.assertEqual([s["name"] for s in out], [f"auto_{i}" for i in range(1, 17)])

    def test_no_structure_at_all_degrades_to_one_chunk(self):
        text = _pad("just plain unstructured filler prose with no markers whatsoever.", 20)
        self.assertGreater(len(text), clozn_sections.LONG_MESSAGE_CHARS)
        out = clozn_sections.auto_chunk_messages(self._messages(text))
        self.assertEqual(len(out), 1)

    def test_non_list_or_empty_input_is_safe(self):
        self.assertEqual(clozn_sections.auto_chunk_messages([]), [])
        self.assertEqual(clozn_sections.auto_chunk_messages(None), [])
        self.assertEqual(clozn_sections.auto_chunk_messages("not a list"), [])


# ======================================================================================================
class AutoChunkPromptTests(unittest.TestCase):
    def test_message_index_is_always_none(self):
        out = clozn_sections.auto_chunk_prompt(HEADER_DOC)
        self.assertEqual(len(out), 3)
        for sec in out:
            for part in sec["parts"]:
                self.assertIsNone(part["message_index"])

    def test_determinism(self):
        self.assertEqual(clozn_sections.auto_chunk_prompt(HEADER_DOC), clozn_sections.auto_chunk_prompt(HEADER_DOC))

    def test_short_prompt_is_one_chunk(self):
        out = clozn_sections.auto_chunk_prompt("short prompt text")
        self.assertEqual(len(out), 1)

    def test_empty_or_non_string_is_safe(self):
        self.assertEqual(clozn_sections.auto_chunk_prompt(""), [])
        self.assertEqual(clozn_sections.auto_chunk_prompt(None), [])

    def test_resolves_against_final_prompt_field(self):
        out = clozn_sections.auto_chunk_prompt(HEADER_DOC)
        run = {"final_prompt": HEADER_DOC, "sections": out}
        text = clozn_sections.resolve(run, out[0]["id"])
        self.assertTrue(text.startswith("# Section One"))


# ======================================================================================================
class SectionsFromNativeTests(unittest.TestCase):
    PROMPT = "0123456789" * 10   # 100 chars

    def test_basic_conversion(self):
        out = clozn_sections.sections_from_native(self.PROMPT, {"rag": {"start": 0, "end": 10}})
        self.assertEqual(len(out), 1)
        sec = out[0]
        self.assertEqual(sec["name"], "rag")
        self.assertEqual(sec["source"], "api")
        self.assertEqual(sec["parts"], [{"message_index": None, "start": 0, "end": 10}])
        self.assertEqual(sec["char_count"], 10)
        self.assertEqual(sec["preview"], self.PROMPT[0:10])

    def test_out_of_range_offsets_are_clamped(self):
        out = clozn_sections.sections_from_native(self.PROMPT, {"tail": {"start": 90, "end": 10_000}})
        self.assertEqual(out[0]["parts"][0]["end"], len(self.PROMPT))

    def test_negative_start_is_clamped_to_zero(self):
        out = clozn_sections.sections_from_native(self.PROMPT, {"neg": {"start": -50, "end": 5}})
        self.assertEqual(out[0]["parts"][0]["start"], 0)

    def test_inverted_or_empty_span_is_dropped_silently(self):
        out = clozn_sections.sections_from_native(self.PROMPT, {
            "inverted": {"start": 10, "end": 5},
            "empty": {"start": 5, "end": 5},
            "good": {"start": 0, "end": 3},
        })
        self.assertEqual([s["name"] for s in out], ["good"])

    def test_non_int_offsets_are_dropped_silently(self):
        out = clozn_sections.sections_from_native(self.PROMPT, {"bad": {"start": "x", "end": 5}})
        self.assertEqual(out, [])

    def test_bad_input_types_are_safe(self):
        self.assertEqual(clozn_sections.sections_from_native(None, {}), [])
        self.assertEqual(clozn_sections.sections_from_native(self.PROMPT, None), [])
        self.assertEqual(clozn_sections.sections_from_native(self.PROMPT, {"x": "not a dict"}), [])


# ======================================================================================================
class MemoryCardSectionsTests(unittest.TestCase):
    def test_locates_card_text_and_emits_a_section(self):
        assembled = [
            {"role": "system", "content": "Preamble. REMEMBER: user prefers tea. Postamble."},
            {"role": "user", "content": "hi"},
        ]
        out = clozn_sections.memory_card_sections(assembled, ["REMEMBER: user prefers tea."])
        self.assertEqual(len(out), 1)
        sec = out[0]
        self.assertEqual(sec["name"], "card_1")
        self.assertEqual(sec["source"], "memory_card")
        self.assertEqual(sec["parts"][0]["message_index"], 0)
        start = sec["parts"][0]["start"]
        self.assertEqual(assembled[0]["content"][start:start + sec["char_count"]],
                         "REMEMBER: user prefers tea.")

    def test_unlocatable_card_is_skipped_silently(self):
        assembled = [{"role": "system", "content": "nothing relevant here"}]
        out = clozn_sections.memory_card_sections(assembled, ["a card text that never appears"])
        self.assertEqual(out, [])

    def test_names_track_input_order_not_document_order(self):
        assembled = [{"role": "system", "content": "SECOND CARD TEXT then FIRST CARD TEXT"}]
        out = clozn_sections.memory_card_sections(assembled, ["FIRST CARD TEXT", "SECOND CARD TEXT"])
        names = [s["name"] for s in out]
        self.assertEqual(names, ["card_1", "card_2"])

    def test_missing_assembled_messages_or_cards_is_safe(self):
        self.assertEqual(clozn_sections.memory_card_sections(None, ["x"]), [])
        self.assertEqual(clozn_sections.memory_card_sections([{"role": "system", "content": "x"}], []), [])
        self.assertEqual(clozn_sections.memory_card_sections([{"role": "system", "content": "x"}], None), [])


# ======================================================================================================
class ResolveTests(unittest.TestCase):
    def test_resolves_by_id_or_by_name(self):
        run = {
            "messages": [{"role": "system", "content": "hello world"}],
            "sections": [{"id": "sec_greeting", "name": "greeting",
                          "parts": [{"message_index": 0, "start": 0, "end": 5}]}],
        }
        self.assertEqual(clozn_sections.resolve(run, "sec_greeting"), "hello")
        self.assertEqual(clozn_sections.resolve(run, "greeting"), "hello")

    def test_multi_part_sections_join_in_order(self):
        run = {
            "messages": [{"role": "system", "content": "AAAA"}, {"role": "assistant", "content": "BBBB"}],
            "sections": [{"id": "sec_x", "name": "x", "parts": [
                {"message_index": 0, "start": 0, "end": 4},
                {"message_index": 1, "start": 0, "end": 4},
            ]}],
        }
        self.assertEqual(clozn_sections.resolve(run, "sec_x"), "AAAABBBB")

    def test_missing_section_returns_empty_string(self):
        self.assertEqual(clozn_sections.resolve({"messages": [], "sections": []}, "nope"), "")
        self.assertEqual(clozn_sections.resolve({}, "nope"), "")
        self.assertEqual(clozn_sections.resolve(None, "nope"), "")

    def test_never_raises_on_malformed_run(self):
        run = {"messages": "not a list", "sections": [{"id": "a", "parts": [{"message_index": 0,
                                                                              "start": 0, "end": 5}]}]}
        self.assertEqual(clozn_sections.resolve(run, "a"), "")


# ======================================================================================================
class StoreRoundTripTests(unittest.TestCase):
    def setUp(self):
        self._original_runs_dir = runlog.RUNS_DIR
        import tempfile
        self._temp = tempfile.TemporaryDirectory(prefix="clozn-sections-test-")
        runlog.RUNS_DIR = self._temp.name

    def tearDown(self):
        runlog.RUNS_DIR = self._original_runs_dir
        self._temp.cleanup()

    def test_record_persists_sections_and_get_run_returns_them(self):
        manifest = [{"id": "sec_rag_context", "name": "rag_context", "source": "auto",
                    "parts": [{"message_index": 0, "start": 0, "end": 4}],
                    "char_count": 4, "preview": "text"}]
        rid = runlog.record(source="cli", messages=[{"role": "user", "content": "text"}],
                            response="ok", sections=manifest)
        self.assertIsNotNone(rid)
        rec = runlog.get_run(rid)
        self.assertEqual(rec["sections"], manifest)

    def test_no_sections_field_when_none_or_empty(self):
        rid_none = runlog.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="ok")
        rid_empty = runlog.record(source="cli", messages=[{"role": "user", "content": "hi"}],
                                  response="ok", sections=[])
        self.assertNotIn("sections", runlog.get_run(rid_none))
        self.assertNotIn("sections", runlog.get_run(rid_empty))


# ======================================================================================================
class _FakeChatSubstrate:
    """Mimics EngineSubstrate's/QwenSubstrate's .chat() surface just enough for openai.py's try_post."""

    def __init__(self):
        self.seen_messages = None
        self.reply = "the reply"
        self.raise_on_chat = None

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self.seen_messages = messages
        if self.raise_on_chat:
            raise self.raise_on_chat
        return self.reply

    def last_finish_reason(self):
        return "stop"


class _FakeHandler:
    def __init__(self):
        self.headers = {"User-Agent": "unittest"}
        self.logged_calls = []
        self.json_calls = []

    def _json(self, code, obj, extra_headers=None):
        self.json_calls.append((code, obj))

    def _log_run(self, source, messages, response, model, started, error=None, trace=None,
                mem_out=None, finish_reason=None, finish_reason_fallback=None, sections=None):
        self.logged_calls.append({"messages": messages, "response": response, "error": error,
                                  "sections": sections})
        return "run_test_id"


class OpenAIRouteWiringTests(unittest.TestCase):
    def setUp(self):
        self._old_sub = app.SUB
        self.sub = _FakeChatSubstrate()
        app.SUB = self.sub
        self.handler = _FakeHandler()

    def tearDown(self):
        app.SUB = self._old_sub

    def test_explicit_tag_wins_and_is_stripped_before_reaching_the_substrate(self):
        body = {"messages": [
            {"role": "system", "content": "RAG CONTEXT BLOCK", "clozn_section": "rag_context"},
            {"role": "user", "content": "the real question"},
        ]}
        handled = openai_routes.try_post(self.handler, "/v1/chat/completions", body)
        self.assertTrue(handled)
        # the substrate/engine never saw the extension field
        for m in self.sub.seen_messages:
            self.assertNotIn("clozn_section", m)
        # the run got an explicit "api" section, not an auto one
        manifest = self.handler.logged_calls[-1]["sections"]
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["name"], "rag_context")
        self.assertEqual(manifest[0]["source"], "api")
        # the reply itself is unaffected
        self.assertEqual(self.json_reply()["choices"][0]["message"]["content"], "the reply")

    def test_backward_compat_no_tags_yields_an_auto_manifest_and_unchanged_reply(self):
        body = {"messages": [
            {"role": "system", "content": HEADER_DOC},
            {"role": "user", "content": "the real question"},
        ]}
        handled = openai_routes.try_post(self.handler, "/v1/chat/completions", body)
        self.assertTrue(handled)
        manifest = self.handler.logged_calls[-1]["sections"]
        self.assertTrue(manifest)
        self.assertTrue(all(s["source"] == "auto" for s in manifest))
        self.assertEqual(self.json_reply()["choices"][0]["message"]["content"], "the reply")

    def test_plain_client_with_no_system_message_gets_an_empty_manifest_not_an_error(self):
        body = {"messages": [{"role": "user", "content": "just a question"}]}
        handled = openai_routes.try_post(self.handler, "/v1/chat/completions", body)
        self.assertTrue(handled)
        self.assertEqual(self.handler.logged_calls[-1]["sections"], [])
        self.assertEqual(self.json_reply()["choices"][0]["message"]["content"], "the reply")

    def test_chat_failure_still_logs_whatever_manifest_was_built(self):
        self.sub.raise_on_chat = RuntimeError("boom")
        body = {"messages": [
            {"role": "system", "content": "ctx", "clozn_section": "rag"},
            {"role": "user", "content": "q"},
        ]}
        handled = openai_routes.try_post(self.handler, "/v1/chat/completions", body)
        self.assertTrue(handled)
        self.assertEqual(self.handler.logged_calls[-1]["sections"][0]["name"], "rag")
        self.assertEqual(self.json_calls_status(), 502)

    def json_reply(self):
        return self.handler.json_calls[-1][1]

    def json_calls_status(self):
        return self.handler.json_calls[-1][0]


# ======================================================================================================
class _StreamSubstrate:
    def chat_stream(self, messages, max_new, mem_out=None, lens=None, on_frame=None):
        yield "hi"

    def last_finish_reason(self):
        return "stop"

    def last_stream_trace(self):
        return []


class _StreamHandler:
    def __init__(self):
        import io
        self.wfile = io.BytesIO()
        self.headers = {}
        self.logged_calls = []

    def send_response(self, code):
        pass

    def send_header(self, key, value):
        pass

    def end_headers(self):
        pass

    def _log_run(self, *args, **kwargs):
        self.logged_calls.append(kwargs)
        return "run_stream_id"


class SSEWiringTests(unittest.TestCase):
    def test_sse_chat_threads_sections_through_to_log_run(self):
        old_sub = app.SUB
        app.SUB = _StreamSubstrate()
        handler = _StreamHandler()
        manifest = [{"id": "sec_x", "name": "x", "source": "auto", "parts": [], "char_count": 0,
                    "preview": ""}]
        try:
            sse.sse_chat(handler, [{"role": "user", "content": "hi"}], 8, "m", sections=manifest)
        finally:
            app.SUB = old_sub
        self.assertEqual(handler.logged_calls[0]["sections"], manifest)


# ======================================================================================================
class LogRunMemoryCardWiringTests(unittest.TestCase):
    """Exercises clozn.server.app's _log_run directly (built via app.make_handler(), same pattern
    test_runtime_architecture.py's raw_gateway_request uses) to prove deliverable 5's wiring: memory cards
    that rode a turn get folded into the section manifest regardless of which route called _log_run.
    facts_mode is force-disabled and every applied card's id is left None (skipping memory_cards.
    bump_usage) so this never touches the real ~/.clozn memory-card/facts stores on the dev machine."""

    def setUp(self):
        self._original_runs_dir = runlog.RUNS_DIR
        import tempfile
        self._temp = tempfile.TemporaryDirectory(prefix="clozn-sections-logrun-test-")
        runlog.RUNS_DIR = self._temp.name
        self._old_sub = app.SUB
        app.SUB = None   # _sub() resolves to None -> every substrate-dependent branch degrades cleanly

    def tearDown(self):
        runlog.RUNS_DIR = self._original_runs_dir
        app.SUB = self._old_sub
        self._temp.cleanup()

    def _handler(self):
        handler_type = app.make_handler()
        handler = object.__new__(handler_type)
        handler.headers = {"User-Agent": "unittest"}
        return handler

    def test_applied_memory_cards_are_located_and_appended_to_the_manifest(self):
        assembled = [
            {"role": "system", "content": "Preamble text. REMEMBER: user prefers dark mode. More text."},
            {"role": "user", "content": "hi"},
        ]
        mem_out = {
            "mode": "prompt",
            "applied": [{"id": None, "text": "REMEMBER: user prefers dark mode."}],
            "assembled_messages": assembled,
        }
        caller_manifest = [{"id": "sec_explicit", "name": "explicit", "source": "api",
                           "parts": [{"message_index": 0, "start": 0, "end": 4}],
                           "char_count": 4, "preview": "Prea"}]
        handler = self._handler()
        with mock.patch("clozn.memory.facts_mode.enabled", return_value=False):
            rid = handler._log_run("openai_api", [{"role": "user", "content": "hi"}], "reply", "model",
                                   time.time(), mem_out=mem_out, sections=caller_manifest)
        self.assertIsNotNone(rid)
        rec = runlog.get_run(rid)
        names = {s["name"]: s for s in rec["sections"]}
        self.assertIn("explicit", names)
        self.assertIn("card_1", names)
        self.assertEqual(names["card_1"]["source"], "memory_card")
        self.assertEqual(names["card_1"]["parts"][0]["message_index"], 0)

    def test_no_cards_applied_leaves_the_callers_manifest_untouched(self):
        caller_manifest = [{"id": "sec_explicit", "name": "explicit", "source": "api",
                           "parts": [{"message_index": 0, "start": 0, "end": 2}],
                           "char_count": 2, "preview": "hi"}]
        handler = self._handler()
        with mock.patch("clozn.memory.facts_mode.enabled", return_value=False):
            rid = handler._log_run("openai_api", [{"role": "user", "content": "hi"}], "reply", "model",
                                   time.time(), sections=caller_manifest)
        rec = runlog.get_run(rid)
        self.assertEqual(rec["sections"], caller_manifest)

    def test_internalized_mode_has_no_assembled_block_so_no_card_sections_are_added(self):
        # internalized mode never sets assembled_messages (the memory lives in a trained prefix, not
        # visible prompt text) -- memory_card_sections must degrade to [] rather than raising.
        handler = self._handler()
        with mock.patch("clozn.memory.facts_mode.enabled", return_value=False):
            rid = handler._log_run("openai_api", [{"role": "user", "content": "hi"}], "reply", "model",
                                   time.time(), mem_out={})
        rec = runlog.get_run(rid)
        self.assertNotIn("sections", rec)


if __name__ == "__main__":
    unittest.main()
