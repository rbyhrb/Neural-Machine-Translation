TRANS_FILE="out.dev"
REF_FILE="en.dev"

perl "mosesdecoder/scripts/tokenizer/detokenizer.perl" -l en < "$TRANS_FILE" > "$TRANS_FILE.detok"
PARAMS=("-tok" "intl" "-l" "de-en" "$REF_FILE")
sacrebleu "${PARAMS[@]}" < "$TRANS_FILE.detok"
