# Análise de Sentimento e NER em Reviews de E-commerce (PT-BR)

Classificação de sentimento em avaliações de produtos em português, por fine-tuning do BERTimbau, com um módulo complementar de reconhecimento de entidades nomeadas.

## Resultado

| Métrica | Valor |
| --- | --- |
| Acurácia global | **76,9%** |
| Precisão — Negativo | 87% |
| Recall — Negativo | 83% |
| Precisão — Positivo | 81% |
| Recall — Positivo | 81% |
| Precisão — Neutro | 63% |

Avaliado em 900 amostras de teste. A classe Neutro é a mais difícil, como esperado: ela se confunde com positivos e negativos brandos.

## O problema

Avaliações de e-commerce chegam como texto livre e nota de 1 a 5. A nota sozinha não diz o que o cliente reclamou, e ler tudo manualmente não escala. O projeto converte o texto em sentimento classificado e extrai as entidades citadas — marcas, pessoas e locais — para que a reclamação possa ser roteada.

## Como funciona

1. **Carga e limpeza** — leitura do dataset B2W, descarte de textos nulos e conversão das notas em três classes: 1–2 negativo, 3 neutro, 4–5 positivo.
2. **Balanceamento** — subamostragem de 1.500 exemplos por classe, 4.500 no total, para impedir que o modelo se vicie na classe positiva, que é maioria natural.
3. **Fine-tuning** — `neuralmind/bert-base-portuguese-cased` (BERTimbau), 3 épocas, taxa de aprendizado 2e-5, lote de 16, comprimento máximo de 128 tokens.
4. **Avaliação** — relatório de classificação e matriz de confusão.
5. **NER** — pipeline separado com `Davlan/bert-base-multilingual-cased-ner-hrl` para extrair pessoas, organizações e locais.

## Stack

Python · PyTorch · Hugging Face Transformers · Datasets · scikit-learn · pandas · Matplotlib · Seaborn

## Como rodar

```bash
pip install -r requirements.txt
# baixe B2W-Reviews01.csv do Kaggle e coloque na raiz
python b2w_reviews01.py
```

## Limitações

- A classe Neutro tem precisão de 63%; é o ponto fraco do modelo.
- A subamostragem descarta a maior parte do dataset original em troca de equilíbrio e tempo de treino.
- O módulo de NER usa modelo pré-treinado genérico, sem ajuste ao domínio de e-commerce.
- O dataset não é versionado aqui: é público e grande demais para o repositório.

## Autoria

Pedro Henrique Acioly Amaral. Modelos de terceiros: BERTimbau (NeuralMind) e NER multilíngue (Davlan).
