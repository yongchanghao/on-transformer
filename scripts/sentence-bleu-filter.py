from nltk.translate import bleu_score
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--ref', required=True)
parser.add_argument('--hyp', required=True)
parser.add_argument('--src', required=True)

parser.add_argument('--filtered-hyp', required=True)
parser.add_argument('--filtered-src', required=True)
parser.add_argument('--reverse-output', action='store_true', default=False)


args = parser.parse_args()
method = bleu_score.SmoothingFunction().method2

reverse_fn = reversed if args.reverse_output else list

with open(args.ref) as ref_fp, open(args.hyp) as hyp_fp, open(args.src) as src_fp, \
        open(args.filtered_src, 'w', encoding='utf8') as f_src_fp, \
        open(args.filtered_hyp, 'w', encoding='utf8') as f_hyp_fp:
    for ref, hyp, src in zip(ref_fp, hyp_fp, src_fp):
        ref_tokens = list(reverse_fn(ref.strip().lower().split()))
        hyp_tokens = list(reverse_fn(hyp.strip().lower().split()))
        score = bleu_score.sentence_bleu(
            [ref_tokens], hyp_tokens,
            smoothing_function=method
        ) * 100
        if score > 5.0:
            src_tokens = list(reverse_fn(src.strip().split()))
            hyp_tokens = list(reverse_fn(hyp.strip().split()))
            src_sentence = " ".join(src_tokens)
            hyp_sentence = " ".join(hyp_tokens)
            f_src_fp.write(src_sentence + '\n')
            f_hyp_fp.write(hyp_sentence + '\n')


