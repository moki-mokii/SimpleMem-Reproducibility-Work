# SimpleMem Reproducibility Work

This FTEC5660 experiment measures how retrieval reflection affects answer quality, latency, and API usage in SimpleMem v0.1.0 (upstream commit `7da777f56a15db81bb261d296c89cad5915e8d67`). After initial hybrid retrieval, the reflection-enabled condition allows up to two rounds of evidence checking and supplemental search. The reflection-disabled condition sets `enable_reflection=False`. Both conditions use the same query planning, hybrid retrieval, answer generation, and scoring procedures.

The evaluation uses [120 fixed LoCoMo questions](experiments/reflection/questions.json) across three conversations and four question categories. Both conditions use a shared memory snapshot for each conversation. Three rounds produce 360 paired comparisons and 720 answer evaluations. The language model is DeepSeek `deepseek-flash` with thinking disabled. Embeddings are computed locally with Qwen3-Embedding-0.6B.

## Installation

The following commands use Git, Windows PowerShell, and Python 3.12. Clone the repository and run the remaining commands from its root. The runner uses Git history to verify the source against the pinned SimpleMem commit.

```powershell
git clone https://github.com/moki-mokii/SimpleMem-Reproducibility-Work.git
Set-Location SimpleMem-Reproducibility-Work
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0+cpu"
python -m pip install -r experiments/reflection/requirements.txt
python -m pip check
```

## Dataset and embedding model

Download the [pinned LoCoMo dataset](https://raw.githubusercontent.com/snap-research/locomo/cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc/data/locomo10.json) and the [pinned Qwen3-Embedding-0.6B checkpoint](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3).

```powershell
New-Item -ItemType Directory -Force test_results/inputs
Invoke-WebRequest 'https://raw.githubusercontent.com/snap-research/locomo/cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc/data/locomo10.json' -OutFile test_results/inputs/locomo10.json
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --include '*.json' 'merges.txt' 'model.safetensors' --local-dir test_results/inputs/qwen3-embedding
```

## Run the evaluation

Create a local `.env` file in the repository root and replace the placeholder with your DeepSeek API key. Git ignores this file.

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
```

Run the evaluation and generate the summary. If `DEEPSEEK_API_KEY` is already set in the process environment, omit `--key-file .env`.

```powershell
python experiments/reflection/run.py `
    --dataset test_results/inputs/locomo10.json `
    --embedding test_results/inputs/qwen3-embedding `
    --output test_results/reflection `
    --key-file .env `
    --mode all
python experiments/reflection/summarize.py test_results/reflection
```

To resume an interrupted run, repeat the evaluation command with the same code, inputs, and output directory. The runner skips completed questions.

## Results

The summarizer writes `test_results/reflection/summary.json` with four-category macro token F1, scores by category and conversation, paired differences by round, latency, HTTP attempts, and token usage. The main analysis assigns F1 zero to answers that cannot be scored. The summary also reports F1 for pairs in which both conditions succeeded.

See the [measured results and examples](experiments/reflection/RESULTS.md) and the [machine-readable summary](experiments/reflection/summary.json) from the completed run.

Read the [reproducibility report (PDF)](experiments/reflection/SimpleMem-Reproducibility-Work-Report.pdf).

View the [presentation slides (PDF)](experiments/reflection/SimpleMem-Reproducibility-Work-Presentation.pdf).

## Upstream documentation

For SimpleMem's architecture and general usage, see the [original SimpleMem README](https://github.com/aiming-lab/SimpleMem/blob/7da777f56a15db81bb261d296c89cad5915e8d67/README.md).
