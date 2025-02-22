# Documentação do Projeto: iFood Dev Week - Análise de NPS e Análise de Sentimentos

**1. Título do Projeto:** iFood Dev Week - Análise de NPS e Análise de Sentimentos

**2. Visão Geral do Projeto:**

*   **Descrição:** Este projeto foi desenvolvido durante a iFood Dev Week e se concentra na análise de dados de Net Promoter Score (NPS) e na realização de análise de sentimentos em comentários de feedback de usuários. Ele demonstra o uso do Google Colab para manipulação de dados, visualização e integração com APIs externas (OpenAI).

*   **Objetivo:** Os principais objetivos do projeto são:
    *   Calcular o NPS com base no feedback do usuário.
    *   Visualizar os resultados do NPS usando um gráfico personalizado.
    *   Analisar os comentários de feedback do usuário para entender o sentimento subjacente usando técnicas de IA.
*   **Tecnologias:**
    *   Google Colab: Para execução de código e desenvolvimento de projetos.
    *   Python: A principal linguagem de programação usada.
    *   Pandas: Para manipulação e análise de dados.
    *   Matplotlib: Para criar visualizações de dados.
    *   OpenAI API: Para realizar análise de sentimentos usando o modelo GPT-4.
    *   Gdown: Para baixar o arquivo de dados de um link do Google Drive.
    *   IPython + Gemini AI: Para documentação em markdown.

**3. Estrutura do Projeto:**

*   **Estrutura do Notebook:** O projeto está organizado em um único notebook do Google Colab, dividido em seções distintas:
    *   **Dia 1: Introdução ao Google Colab e Projeto NPS:**
        *   Carregamento de Dados: Baixando o arquivo `feedbacks.csv` do Google Drive usando a biblioteca `gdown`.
        *   Cálculo de NPS (Imperativo): Uma abordagem sequencial para calcular o NPS.
        *   Cálculo de NPS (Funcional): Usando funções para separar o código e melhorar a legibilidade.
        *   Cálculo de NPS (Orientado a Objetos): Usando classes para abstração de dados e métodos.
    *   **Dia 2: Visualização de Dados com Python:**
        *   Criando uma Visualização de NPS: Gerando um gráfico de barras horizontais para representar o valor do NPS, categorizado em zonas predefinidas.
    *   **Dia 3: Análise de Sentimentos com IA:**
        *   Configurando a API OpenAI: Instalando a biblioteca `openai` e configurando a chave da API.
        *   Análise de Sentimentos: Usando o modelo GPT-4 para realizar análise de sentimentos nos comentários dos usuários.

**4. Descrição dos Dados:**

*   **Dataset:** `feedbacks.csv`
    *   **Fonte:** Baixado de um link do Google Drive.
    *   **Formato:** Arquivo CSV, delimitado por ponto e vírgula.
    *   **Colunas:**
        *   `nota`: Representa a pontuação de feedback do usuário.
        *   `comentario`: Contém os comentários de forma livre do usuário.

**5. Explicação do Código:**

#### **Dia 1: Cálculo do NPS**
    *   **Carregamento de Dados:**

```python
 import gdown import pandas as pd
 file_id = '1_nyLNsT55X37Y0KDBXVXzSYsX-SoSqZi'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', 'feedbacks.csv')

    dados = pd.read_csv('/content/feedbacks.csv', delimiter=';')
 
```
*   **Descrição:** O código começa importando as bibliotecas necessárias, `gdown` e `pandas`. Em seguida, ele baixa o arquivo `feedbacks.csv` do link do Google Drive, usando o ID do arquivo. Finalmente, ele lê o arquivo CSV em um DataFrame pandas chamado `dados`.


    *   **Abordagem Imperativa:**

```python
notas = dados['nota']
def calcular_nps(notas):
      detratores = notas.apply(lambda nota: nota <= 6).sum()
      promotores = notas[notas >= 9].count()

      return (promotores - detratores) / len(notas) * 100

    notas = dados['nota']

    nps = calcular_nps(notas)

    print(nps)

```
*   **Descrição:** Este código calcula o NPS usando uma abordagem funcional. Ele define uma função `calcular_nps` que recebe as pontuações de feedback como entrada. Dentro da função, ele usa funções pandas como `apply`, `sum` e `count` para contar detratores e promotores. O NPS é calculado e retornado pela função.

    *   **Abordagem Orientada a Objetos:**
```python
class Feedback: def init(self, nota, comentario): self.nota = nota self.comentario = comentario
class AnalisadorFeedback:
      def __init__(self, feedbacks):
        self.feedbacks = feedbacks

      def calcular_nps(self):
        detratores = sum(1 for feedback in self.feedbacks if feedback.nota <= 6)
        promotores = sum(1 for feedback in self.feedbacks if feedback.nota >= 9)

        return (promotores - detratores) / len(self.feedbacks) * 100

    feedbacks = dados.apply(lambda linha: Feedback(linha['nota'], linha['comentario']), axis=1)

    analisador = AnalisadorFeedback(feedbacks)
    nps = analisador.calcular_nps()

    print(nps)

``` 
*   **Descrição:** Este código calcula o NPS usando uma abordagem orientada a objetos. Ele define duas classes: `Feedback` para representar um único feedback e `AnalisadorFeedback` para analisar uma lista de feedbacks. A classe `AnalisadorFeedback` tem um método `calcular_nps` que calcula o NPS. O código então cria uma lista de objetos `Feedback` e um objeto `AnalisadorFeedback`, e chama o método `calcular_nps` para obter o NPS.

####   **Dia 2: Visualização de NPS**

```python
import matplotlib.pyplot as plt import matplotlib.patches as mpatches

NPS_ZONAS =   ['Crítico', 'Aperfeiçoamento', 'Qualidade', 'Excelência']
NPS_VALORES = [-100, 0, 50, 75, 100]
NPS_CORES =   ['#FF595E', '#FFCA3A', '#8AC926', '#1982C4']

def criar_grafico_nps(nps):
  fig, ax = plt.subplots(figsize=(10, 2))

  for i, zona in enumerate(NPS_ZONAS):
    ax.barh([0], width=NPS_VALORES[i+1]-NPS_VALORES[i], left=NPS_VALORES[i], color=NPS_CORES[i])

  ax.barh([0], width=1, left=nps, color='black')
  ax.set_yticks([])
  ax.set_xlim(-100, 100)
  ax.set_xticks(NPS_VALORES)

  plt.text(nps, 0, f'{nps:.2f}', ha='center', va='center', color='white', bbox=dict(facecolor='black'))

  patches = [mpatches.Patch(color=NPS_CORES[i], label=NPS_ZONAS[i]) for i in range(len(NPS_ZONAS))]
  plt.legend(handles=patches, bbox_to_anchor=(1,1))

  plt.title('Gráfico de NPS da iFood Dev Week')

  plt.show()

criar_grafico_nps(nps)

``` 
*   **Descrição:** Esta seção usa a biblioteca `matplotlib` para gerar um gráfico de barras horizontais para visualizar o valor do NPS. A função `criar_grafico_nps` recebe o NPS calculado como entrada e cria o gráfico. Ele usa zonas codificadas por cores para representar diferentes níveis de NPS e coloca uma barra preta para marcar o valor exato do NPS.

####   **Dia 3: Análise de Sentimentos**

```python
!pip install openai==0.28.1
import openai

openai.api_key = openai_api_key

def analisar_sentimentos(feedbacks):

  comentarios_formatados = "\n".join([f"- Nota {feedback.nota}! {feedback.comentario}" for feedback in feedbacks])
  prompt = f"""
            Sintetize uma análise geral sobre os seguintes comentário:
            {comentarios_formatados}"""

  respostaAPI = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
      {
          "role": "system",
          "content": "Você é um modelo de análise de sentimentos com foco em feedbacks sobre experiências educacionais."
      },
      {
          "role": "user",
          "content": prompt
      }
    ]
  )
  return respostaAPI.choices[0].message.content

insigths = analisar_sentimentos(feedbacks)
print(insigths)

```
*   **Descrição:** Esta linha instala a biblioteca `openai`, versão 0.28.1, necessária para interagir com a API OpenAI.

*   **Descrição:** Primeira linha instala a biblioteca `openai`, versão 0.28.1, necessária para interagir com a API OpenAI.Esta seção manipula a análise de sentimentos usando a API OpenAI.
        *   **Autenticação:** O código começa exigindo que o usuário insira sua chave de API OpenAI, que será usada para autenticar com a API OpenAI. Ele não salva a chave de API como uma variável.
        *   **Função `analisar_sentimentos`:** Esta função recebe uma lista de objetos `Feedback` como entrada.
            *   **Geração de Prompt:** Ela cria uma string formatada dos comentários de feedback para ser usada como um prompt para o modelo GPT-4.
            *   **Chamada da API:** Ela faz uma chamada para a API OpenAI, especificando o modelo `gpt-4` e fornecendo um array de mensagens. O array inclui:
                *   Uma mensagem de sistema, que diz ao modelo que ele deve atuar como uma ferramenta de análise de sentimentos focada em feedback educacional.
                *   Uma mensagem de usuário, que contém o prompt com o feedback do usuário.
            *   **Manipulação da Resposta:** Ela processa a resposta da API e extrai o sentimento analisado da primeira escolha e da primeira mensagem.
        * **Chamada da função**: Finalmente, a função `analisar_sentimentos` é chamada com a lista de `feedbacks` e imprime a análise retornada.

**6. Configuração e Instalação:**

1.  **Google Colab:** Acesse o notebook do Google Colab fornecido.
2.  **Instalar Bibliotecas:** O código deve rodar sem erros, já que já está configurado no notebook Colab, mas se precisar executar este projeto fora do Colab, instale as bibliotecas necessárias:

```python
pip install pandas gdown matplotlib openai==0.28.1

``` 
3.  **Chave de API OpenAI:** Para executar a análise de sentimentos, siga estes passos:
    *   Acesse o site da OpenAI.
    *   Faça login ou crie uma conta.
    *   Navegue até a seção "API Keys".
    *   Crie uma nova chave de API.
    *   Substitua o espaço reservado `TODO` no código pela sua chave.

**7. Uso:**

1.  **Executar o Código:** Execute cada célula sequencialmente no notebook Colab.
2.  **Visualizar:** Observe o gráfico de NPS.
3.  **Análise de Sentimentos:** A saída da análise de sentimentos será impressa no console.

**8. Conclusão:**

Este documento descreve o projeto iFood Dev Week, que abrange cálculos de NPS, visualização de dados e análise de sentimentos. Ele mostra uma variedade de técnicas de ciência de dados usando Python, Google Colab e a API OpenAI. 

O projeto bem como os resultados de execução do mesmo, podem ser localizados no seguinte [REPOSITÓRIO.](https://github.com/paulo-freitas-junior/ifood-dev-week2023)
