# 🧠 Portuguese Hate Speech Detector

Modelo de classificação de discurso de ódio em português, desenvolvido com **Transformers (Hugging Face)**.  
Este projeto realiza **fine-tuning de um modelo BERT** sobre múltiplos datasets de linguagem ofensiva, com o objetivo de detectar comentários de ódio e toxicidade em texto.

---

## 🚀 Visão Geral

O **Portuguese Hate Speech Detector** combina dois datasets públicos:
- [`manueltonneau/portuguese-hate-speech-superset`](https://huggingface.co/datasets/manueltonneau/portuguese-hate-speech-superset)
- [`franciellevargas/HateBR`](https://huggingface.co/datasets/franciellevargas/HateBR)

O modelo base é o **[`ruanchaves/bert-large-portuguese-cased-hatebr`](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-hatebr)**, ajustado com pesos de classe para lidar com desbalanceamento de dados.  
O objetivo é detectar se um texto contém **discurso de ódio (classe 1)** ou **linguagem neutra (classe 0)**.

---

## **Clonar o repositório**

```bash
git clone https://github.com/taismoreira2023/hate-speech-pt-classifier.git
cd hate-speech-pt-classifier
```
## **Criar o ambiente virtual**

### **Windows (PowerShell)**

```powershell
.\venv.ps1
```

### **Linux / macOS**

```bash
./venv.sh
```

---

## ⚙️ Tecnologias Utilizadas

- 🧩 [Hugging Face Transformers](https://huggingface.co/transformers)
- 📚 [Datasets](https://huggingface.co/docs/datasets)
- 🧮 [PyTorch](https://pytorch.org/)
- 📈 [Evaluate (Hugging Face)](https://huggingface.co/docs/evaluate)
- 🧰 [Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
- 💾 GPU com CUDA (opcional, mas recomendável)

---
## 🧠 Treinamento
Para executar o treinamento completo:

```bash
python script/classificacao_discurso_odio.py
```

O script realiza:
1. Carregamento e limpeza dos datasets.  
2. Tokenização com `AutoTokenizer`.  
3. Concatenação de múltiplos conjuntos anotados.  
4. Aplicação de pesos de classe para lidar com desbalanceamento.  
5. Treinamento e avaliação usando a API do `Trainer`.

### Parâmetros principais:
- `batch_size = 8`
- `gradient_accumulation_steps = 8`
- `epochs = 3`
- `fp16 = True` (treinamento em meia precisão)
- `metric_for_best_model = "f1"`

---

## 📊 Métricas

As métricas utilizadas para avaliação são:
- **Accuracy**  
- **F1 (macro)** – métrica principal usada para salvar o melhor modelo.

Exemplo de saída (simulada):

| Época | Accuracy | F1 (macro) |
|--------|-------------|--------------|
| 1 | 0.804815 | 0.725156 |
| 2 | 0.838523 | 0.753326 |
| 3 | 0.852220 | 0.761679 ✅ (melhor modelo) |

---
##**Executar a inferência (classificação de texto)**

Criado arquivo `predict.py` com o conteúdo:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "results1/classificador_odio.model/checkpoint-1755"  # ajuste para o seu caminho

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

example = "Esse trabalho é uma porcaria inútil"
inputs = tokenizer(example, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

pred = torch.argmax(outputs.logits, dim=-1).item()

print(f"Texto: {example}")
print(f"Classe prevista: {pred}")
# 0 = neutro, 1 = discurso de ódio
```

Execute:

```bash
python predict.py
```

---
🧾 Licença
Este projeto é distribuído sob a licença MIT.
Você é livre para usar, modificar e distribuir, desde que mantenha os créditos.
