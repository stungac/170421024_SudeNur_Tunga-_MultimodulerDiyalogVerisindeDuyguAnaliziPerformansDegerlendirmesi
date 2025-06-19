for SPLIT in train valid test; do
    python -m multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "../../data/processed/meld-text/$SPLIT.input0" \
        --outputs "../../data/processed/meld-text/$SPLIT.input0.bpe" \
        --workers 60 \
        --keep-empty
done 

fairseq-preprocess \
    --only-source \
    --trainpref "../../data/processed/meld-text/train.input0.bpe" \
    --validpref "../../data/processed/meld-text/valid.input0.bpe" \
    --destdir "../../data/processed/meld-text/meld-text-bin/input0" \
    --workers 60 \
    --srcdict ../../src/RoBERTa/roberta-large/dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "../../data/processed/meld-text/train.label" \
    --validpref "../../data/processed/meld-text/valid.label" \
    --destdir "../../data/processed/meld-text/meld-text-bin/label" \
    --workers 60