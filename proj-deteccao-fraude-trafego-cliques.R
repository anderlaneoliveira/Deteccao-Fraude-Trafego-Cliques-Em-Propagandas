#Projeto Data Science - Aplicação de Machine Learning para Detecção de Fraudes
#Autora: Anderlane Oliveira
#Data: "Agosto/2021"


# Objetivo

### criar um modelo de aprendizado de máquina que possa prever se um usuário fará o download de um aplicativo depois de clicar em um anúncio de aplicativo para dispositivos móveis.


# Etapas do Projeto

### 1.Carregamento e vizualização dos dados obtidos
### 2. Informações preliminares sobre os dados
### 3. Transformação e Manipulação dos Dados
### 4. Análise Exploratória de Dados
### 5. Modelagem Preditiva Sem Balanceamento do Dataset
### 6. Modelagem Preditiva após Balanceamento do Dataset
### 7. Avaliação do Modelo Preditivo


# Desenvolvimento

# 1. Carregamento e vizualização dos dados obtidos

### Dataset disponível no site Kaggle:
### https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data


## Descrição das variáveis:

#### ip: endereço ip do clique.
#### app: id do app para marketing.
#### device: ID do tipo de dispositivo do telefone celular do usuário (por exemplo, iphone 6 plus, iphone 7, huawei mate 7 etc.)
#### os: id da versão do sistema operacional do telefone celular do usuário
#### channel: id do canal do editor de anúncios para celular
#### click_time: carimbo de data / hora do clique (UTC)
#### attributed_time: se o usuário baixar o aplicativo depois de clicar em um anúncio, é o momento do download do aplicativo
#### is_attributed: o destino que deve ser previsto, indicando que o aplicativo foi baixado

#### Observação: As variáveis ip, app, device, os e channel são codificados.


# Carregando os pacotes
library(dplyr) # Manipular os dados
library(corrplot) # Criar matrix de correlação
library(ggplot2) # Criar gráficos
library(class) # Criar modelo preditivo


# Carregando o dataset
dados <- read.csv("train_sample.csv")
head(dados)



# 2. Informações preliminares sobre os dados

## Tipo de dados
str(dados)

## Dimensão dos dados
dim(dados)

## Classe
class(dados)

## Sumário
summary(dados)

## Verificando dados missing no dataset
sum(is.na(dados))

# Utilizando Tabelas de Contigência para verificação preliminar da variável alvo
table(dados$is_attributed)



# 3. Transformação e Manipulação dos Dados

## Formatando colunas codificadas como variáveis categóricas

# Função para transformar variáveis para o tipo fator
transf_factor = function(df, variavel){
  for (variavel in variavel){
    df[[variavel]] <- as.factor(df[[variavel]])
  }
  return(df)
}

# Lista de variáveis
cat.var <- c('ip', 'app', 'device', 'os', 'channel', 'is_attributed')

# Transformando os dados
dados <- transf_factor(dados, cat.var)


## Formatando as colunas de data

# Formatando variáveis data com POSIXct
dados$click_time <- as.POSIXct(dados$click_time, format = "%Y-%m-%d %H:%M:%S")
dados$attributed_time <- as.POSIXct(dados$click_time, format = "%Y-%m-%d %H:%M:%S")

# Vizualização do resultado após transformações realizadas
str(dados)
summary(dados)



# 4. Análise Exploratória dos Dados


## Vizualização dos dados em diferentes perspectivas

# Verificando o quantitativo por IP
dados %>% 
  filter(is_attributed == 0) %>% 
  group_by(ip) %>%
  summarise(total = n(), perc_total = n() / 99773 * 100) %>% 
  arrange(desc(total))

# Verificando o quantitativo por APP
dados %>% 
  filter(is_attributed == 0) %>% 
  group_by(app) %>%
  summarise(total = n(), perc_total = n() / 99773 * 100) %>% 
  arrange(desc(total))

# Verificando o quantitativo pela DATA
dados %>% 
  filter(is_attributed == 0) %>% 
  group_by(click_time) %>%
  summarise(total = n(), perc_total = n() / 99773 * 100) %>% 
  arrange(desc(total))

# Agrupando o quantitativo por APP e IP
dados %>% 
  filter(is_attributed == 0) %>% 
  group_by(app, ip) %>%
  summarise(total = n(), perc_total = n() / 99773 * 100) %>% 
  arrange(desc(total))


## Gráficos

# Correlação entre variáveis
dados1 <- read.csv("train_sample.csv")
var_num <- sapply(dados1, is.numeric)
corr.matrix <- cor(dados1[,var_num])
corrplot(corr.matrix, main="\n\nGráfico de Correlação para Variáveis Numéricas", method="number")


# Histograma
hist(table(dados$os), main = "Histograma dos Sist. Operacionais dos Usuários")
hist(table(dados$device), main = "Histograma dos Disp. Celulares dos Usuários")
hist(table(dados$is_attributed), main = "Hist. dos Total de Aplicativos Baixados")


# Plot
plot(x = dados$channel, ann = FALSE)
title(main = "Gráfico dos Canais de Divulgação Utilizados")


# Plot
plot(x = dados$ip, ann = FALSE)
title(main = "Gráfico dos IP's de Acesso pelos Usuários")


## Distrbuição de dados para Variável Target
## Verificando se há necessidade de balanceamento no dataset


# Total por condição: 0(dowload não realizado) ou 1 (dowload realizado)
target_count <- table(dados$is_attributed)
target_count

# Proporção em percentual
target_perc <- round(prop.table(target_count) * 100, digits = 1)
target_perc



# 5. Modelagem Preditiva Sem o Balanceamento do Dataset

## Criando um subset

df1 <- dados
df1$attributed_time = NULL
df1$click_time = NULL
str(df1)


## Criando os dados de treino e teste

# Particionando os dados em treino e teste

amostra = sample(1:nrow(df1), size = 0.7 * nrow(df1))

dados_treino <- df1[amostra, ]
dados_teste <- df1[-amostra, ]

# Conferindo a proporção
table(df1$is_attributed)
prop.table(table(df1$is_attributed))

# Dimensões dos dados
dim(dados_treino)
dim(dados_teste)

# Criando os labels para os dados de treino e teste
dados_treino_label <- df1[amostra, 6]
dados_teste_label <- df1[-amostra, 6]

# Tamanho dos dados
length(dados_treino_label)
length(dados_teste_label)


# Criando o modelo preditivo
## Modelo Preditivo com o Algoritmo k-Nearest Neighbour Classification
modelo_knn_v1 <- knn(train = dados_treino,
                     test = dados_teste,
                     cl = dados_treino_label,
                     k = 21)



summary(modelo_knn_v1)



# 6. Modelagem Preditiva após Balanceamento do Dataset

## Balanceamento do dataset

#Carregando o pacote ROSE
library(ROSE)

# Efetuando o balanceamento dos dados
# Dados de treino
rose_treino <- ROSE(is_attributed ~ ., data = dados_treino, seed = 1)$data

# Dimensões
dim(rose_treino)
prop.table(table(rose_treino$is_attributed))

# Dados de teste
rose_teste <- ROSE(is_attributed ~ ., data = dados_teste, seed = 1)$data

# Dimensões
dim(rose_teste)
prop.table(table(rose_teste$is_attributed))


## Criando os labels para os dados de treino e teste

# Labels para dados de treino e teste
rose_treino_label <- df1[amostra, 6]
rose_teste_label <- df1[-amostra, 6]

# Tamanho
length(rose_treino_label)
length(rose_teste_label)


## Criando o modelo preditivo

## Modelo Preditivo com Algoritmo k-Nearest Neighbour Classification

# Construindo um modelo de classificação
modelo_knn_v2 <- knn(train = rose_treino,
                     test = rose_teste,
                     cl = rose_treino_label,
                     k = 21)


# Resultados obtidos
summary(modelo_knn_v2)



# 7. Avaliação dos Modelos Preditivos

## Criando uma tabela cruzada dos dados previstos x dados atuais

library(gmodels)
# Tabela Cruzada para o Modelo Preditivo 1
CrossTable(x = dados_teste_label, y = modelo_knn_v1, prop.chisq = FALSE)

# Tabela Cruzada para o Modelo Preditivo 2
CrossTable(x = rose_teste_label, y = modelo_knn_v2, prop.chisq = FALSE)



