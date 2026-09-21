# Lecture 9 — Let's build the GPT Tokenizer — progress

> Agent-maintained. Concept-by-concept status + where he got stuck/broke through.
> Read at the start of a lecture-9 session; update when he asks.
> NOTE: repo transcript numbering: file "09" = the GPT Tokenizer (this lecture),
> file "10" = reproduce GPT-2. He calls the tokenizer "lecture 9". His code folder
> is `my_implementaion/lecture_9/`.

## Session mode he chose
Grok-first: chat through the WHOLE lecture at a high level and re-derive every piece
himself (Socratic nudges, no spilled answers) BEFORE watching the video, THEN implement.
He said "assume I understand up to lecture 8" (GPT + tokenization-as-used-in-L7).

## Anchor from lecture 7 (his strong ground)
- Transformer eats a sequence of integer tokens; each indexes a **token embedding table**
  (was 65 rows in gpt.py) and the **LM head** outputs logits over vocab_size.
- L7 used a naive **character-level** tokenizer (65 chars). That already IS tokenization.
- So vocab_size touches exactly two places in gpt.py: embedding rows + final linear size.

## The ladder we're nudging through (central tension = vocab size ↔ sequence length)
1. Why not feed text directly / why char-level is one extreme (tiny vocab, long seqs).
2. The tradeoff extremes (char vs word/sentence-level) → want a tunable middle.
3. Right starting alphabet: Unicode code points (~150k, unstable) → UTF-8 bytes (256, covers all).
4. Raw bytes → seqs too long → compress with BPE (merge most-frequent adjacent pair, repeat).
5. Implementation: get_stats, merge, train loop, merges dict, compression ratio.
6. Tokenizer = separate stage (own training set, run once), then encode / decode.
7. decode (ids→bytes→utf8, errors="replace" gotcha); encode (bytes→apply merges in learned order).
8. Regex splitting (GPT-2/GPT-4) to forbid merges across categories; tiktoken; special tokens.
9. sentencepiece (Llama) — BPE on code points + byte fallback; vocab-size considerations.
10. Quirks explained: spelling, reverse, arithmetic, non-English, python spaces, trailing
    space, <|endoftext|>, SolidGoldMagikarp (untrained embedding row).

## Status
- Just starting. High-level map given; Socratic derivation in progress.

## Teaching notes carried in
- Reads partially / stops at first snag → front-load, self-contained paragraphs, short.
- Percentages + toy numbers + physical pictures. Origin stories land well.
- He drives logic, pulls unguessable APIs one at a time. Groks before coding.
- Visual/spatial → HTML+SVG in llm_output/ (stdlib python via `py`, not python3.13t).
  Candidate visuals for this lecture: vocab↔seqlen tradeoff curve, the merge "forest",
  UTF-8 byte layout, regex category-splitting.
