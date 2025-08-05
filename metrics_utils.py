import numpy as np
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling


class TextMetricsCalculator:
    """
    Calculates ROUGE-L, SBERT cosine similarity, BERT reconstruction loss,
    and a combined memorization score. Loads SBERT, ROUGE, and BERT once.
    """
    def __init__(
        self,
        sbert_model_name: str = 'all-MiniLM-L6-v2',
        use_rouge: bool = True,
        use_cosine: bool = True,
        use_reconstruction: bool = False,
        bert_model_name_or_path: str = None,
        device: str = None,
        num_masking_passes: int = 1
    ):
        self.use_rouge = use_rouge
        self.use_cosine = use_cosine
        self.use_reconstruction = use_reconstruction
        self.num_masking_passes = num_masking_passes

        if use_rouge:
            self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        if use_cosine:
            self.sbert = SentenceTransformer(sbert_model_name)

        if use_reconstruction:
            if bert_model_name_or_path is None:
                raise ValueError('`bert_model_name_or_path` must be provided for reconstruction.')
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name_or_path)
            self.model = BertForMaskedLM.from_pretrained(bert_model_name_or_path).to(self.device).eval()
            self.collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=0.25
            )

    def compute_full_reconstruction_loss(
        self,
        text: str,
        max_length: int = 256,
        min_tokens: int = 20
    ) -> float:
        """
        Compute BERT MLM reconstruction loss over the full text by:
        1. Tokenizing without truncation.
        2. Splitting into chunks of <= max_length.
        3. Merging a too-small final chunk if < min_tokens.
        4. Padding each chunk to max_length.
        5. Masking 25% and averaging loss over all chunks and passes.
        """
        enc = self.tokenizer(text, return_tensors='pt', truncation=False)
        ids = enc['input_ids'][0]
        mask = enc['attention_mask'][0]
        total_len = ids.size(0)

        # Split into chunks
        chunks = []
        for i in range(0, total_len, max_length):
            end = min(i + max_length, total_len)
            chunks.append({'input_ids': ids[i:end], 'attention_mask': mask[i:end]})

        # Merge small final chunk
        if len(chunks) > 1 and chunks[-1]['input_ids'].size(0) < min_tokens:
            last = chunks.pop()
            prev = chunks.pop()
            vlen = int(prev['attention_mask'].sum().item())
            pids = prev['input_ids'][:vlen]
            pmk = prev['attention_mask'][:vlen]
            mids = torch.cat([pids, last['input_ids']])
            mmk = torch.cat([pmk, last['attention_mask']])
            if mids.size(0) < max_length:
                padl = max_length - mids.size(0)
                pad_ids = torch.full((padl,), self.tokenizer.pad_token_id, dtype=torch.long)
                pad_m = torch.zeros(padl, dtype=torch.long)
                mids = torch.cat([mids, pad_ids])
                mmk = torch.cat([mmk, pad_m])
            chunks.append({'input_ids': mids, 'attention_mask': mmk})

        # Pad chunks to max_length
        proc = []
        for ch in chunks:
            cur = ch['input_ids'].size(0)
            if cur < max_length:
                padl = max_length - cur
                pad_ids = torch.full((padl,), self.tokenizer.pad_token_id, dtype=torch.long)
                pad_m = torch.zeros(padl, dtype=torch.long)
                new_ids = torch.cat([ch['input_ids'], pad_ids])
                new_m = torch.cat([ch['attention_mask'], pad_m])
                proc.append({'input_ids': new_ids, 'attention_mask': new_m})
            else:
                proc.append(ch)

        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for ch in proc:
                for _ in range(self.num_masking_passes):
                    example = {'input_ids': ch['input_ids'], 'attention_mask': ch['attention_mask']}
                    batch = self.collator([example])
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    out = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    total_loss += out.loss.item()
                    count += 1

        return (total_loss / count) if count > 0 else 0.0

    def compute(self, gold_text: str, generated_text: str) -> dict:
        """
        Compute metrics between reference and generation:
          - rougeL
          - cosine_similarity
          - reconstruction_loss (full-sequence)
          - memorization_score
        """
        results = {}

        # 1) ROUGE-L
        if self.use_rouge:
            score = self.scorer.score(gold_text, generated_text)['rougeL']
            results['rougeL'] = score.recall

        # 2) SBERT Cosine Similarity
        if self.use_cosine:
            emb_gold = self.sbert.encode(gold_text, convert_to_tensor=True)
            emb_gen = self.sbert.encode(generated_text, convert_to_tensor=True)
            results['cosine_similarity'] = util.cos_sim(emb_gold, emb_gen).item()

        # 3) BERT Reconstruction Loss
        if self.use_reconstruction:
            recon_loss = self.compute_full_reconstruction_loss(generated_text)
            results['reconstruction_loss'] = recon_loss

        # 4) Memorization Score
        if (
            self.use_reconstruction and
            all(k in results for k in ('reconstruction_loss', 'rougeL', 'cosine_similarity'))
        ):
            inv_loss = 1.0 - np.clip(
                (results['reconstruction_loss'] - 0.01) / (8.0 - 0.01),
                0.0, 1.0
            )
            z = (
                4.0 * inv_loss +
                4.5 * results['rougeL'] +
                1.5 * results['cosine_similarity'] -
                5.0
            )
            results['memorization_score'] = 1.0 / (1.0 + np.exp(-z))

        return results


def compute_text_metrics(
    gold_text: str,
    generated_text: str,
    sbert_model_name: str = 'all-MiniLM-L6-v2',
    use_rouge: bool = True,
    use_cosine: bool = True,
    use_reconstruction: bool = False,
    bert_model_name_or_path: str = None,
    device: str = None,
    num_masking_passes: int = 1
) -> dict:
    """
    One-off wrapper that instantiates TextMetricsCalculator and computes metrics.
    """
    calc = TextMetricsCalculator(
        sbert_model_name=sbert_model_name,
        use_rouge=use_rouge,
        use_cosine=use_cosine,
        use_reconstruction=use_reconstruction,
        bert_model_name_or_path=bert_model_name_or_path,
        device=device,
        num_masking_passes=num_masking_passes
    )
    return calc.compute(gold_text, generated_text)
