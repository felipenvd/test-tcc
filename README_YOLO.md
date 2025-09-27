# Detecção de Lesões e Perdas em Carcaças Bovinas - YOLO

Este projeto utiliza o framework Darknet/YOLO para detecção automática de lesões e perdas em carcaças bovinas.

## Estrutura do Projeto

```
/home/felipe/Projects/tcc/
├── deteccao-de-lesoes-e-perdas-nas-carcacas-bovinas/
│   ├── train/images/   # 720 imagens + anotações (JPG + TXT juntos)
│   ├── valid/images/   # 90 imagens + anotações (JPG + TXT juntos)
│   └── test/images/    # 90 imagens + anotações (JPG + TXT juntos)
├── obj.names           # Nomes das 3 classes
├── obj.data            # Configuração do dataset
├── train.txt           # Lista de imagens de treinamento
├── valid.txt           # Lista de imagens de validação
├── yolov4-custom.cfg   # Configuração da rede neural (customizada para 3 classes)
├── yolov4.conv.137     # Pesos pré-treinados (baixados via setup.sh)
├── backup/             # Diretório onde os modelos treinados são salvos
├── setup.sh            # Script de configuração inicial (EXECUTAR PRIMEIRO)
├── train.sh            # Script para iniciar o treinamento
├── test.sh             # Script para testar o modelo
├── README_YOLO.md      # Este arquivo
└── README_YOLOV4_CONFIG.md  # Explicação detalhada do arquivo .cfg
```

## Classes Detectadas

O modelo foi configurado para detectar 3 tipos de lesões/perdas:

1. **Lesao no quarto traseiro** (ID: 0)
2. **Perda no quarto dianteiro** (ID: 1)
3. **Perda no quarto traseiro** (ID: 2) 

## Dataset

- **Total de imagens**: 900 imagens
- **Treinamento**: 720 imagens (80%)
- **Validação**: 90 imagens (10%)
- **Teste**: 90 imagens (10%)
- **Formato**: YOLO (arquivos .txt com anotações normalizadas)

## ⚡ Setup Inicial (OBRIGATÓRIO)

**ANTES DE FAZER QUALQUER COISA**, execute o script de configuração:

```bash
cd /home/felipe/Projects/tcc
./setup.sh
```

Este script vai:
- ✅ Baixar os pesos pré-treinados (yolov4.conv.137 - 162MB)
- ✅ Criar o diretório backup/
- ✅ Verificar se todos os arquivos necessários existem
- ✅ Mostrar estatísticas do dataset

⚠️ **IMPORTANTE**: O arquivo `yolov4.conv.137` é essencial mas **NÃO** está no Git devido ao tamanho (162MB). O script baixa automaticamente.

## Como Usar

### 1. Setup (PRIMEIRA VEZ)

```bash
./setup.sh
```

### 2. Treinamento

Para iniciar o treinamento do modelo:

```bash
cd /home/felipe/Projects/tcc
./train.sh
```

O treinamento foi configurado para:
- 3000 iterações (otimizado)
- Batch size: 64, Subdivisions: 32
- Cálculo de mAP a partir da iteração 200
- Learning rate: 0.0001 (estável)
- Resolução: 512x512 (economia GPU)
- Salvamento automático dos melhores pesos em `backup/`

### 2. Teste e Avaliação

#### Testar com uma imagem específica:
```bash
./test.sh /caminho/para/imagem.jpg
```

#### Calcular mAP no conjunto de validação:
```bash
./test.sh
```

### 3. Uso Manual do Darknet

#### Comando de treinamento manual:
```bash
darknet detector train obj.data yolov4-custom.cfg yolov4.conv.137 -map -clear
```

#### Testar modelo:
```bash
darknet detector test obj.data yolov4-custom.cfg backup/yolov4-custom_best.weights imagem.jpg
```

#### Calcular mAP:
```bash
darknet detector map obj.data yolov4-custom.cfg backup/yolov4-custom_best.weights
```

## Configurações da Rede

O arquivo `yolov4-custom.cfg` foi customizado para este dataset:

- **Classes**: 3 (ao invés de 80 do COCO)
- **Filters**: 24 nas camadas de saída (fórmula: (classes + 5) × 3)
- **Max batches**: 3000 (1000 × número de classes)
- **Steps**: 2400, 2700 (80% e 90% do max_batches)
- **Burn in**: 200 (warm-up inicial)
- **Learning rate**: 0.0001 (otimizado para estabilidade)
- **Resolução**: 512x512 (economia de GPU)
- **Subdivisions**: 32 (balanço qualidade/memória)
- **Batch size**: 64
- **Subdivisions**: 16

## Monitoramento do Treinamento

Durante o treinamento, você verá:

- **Loss**: Deve diminuir gradualmente
- **Avg loss**: Média das perdas
- **Rate**: Taxa de aprendizado atual
- **mAP**: Precisão média calculada periodicamente

## Arquivos de Saída

Após o treinamento, você encontrará na pasta `backup/`:

- `yolov4-custom_last.weights` - Último modelo salvo
- `yolov4-custom_best.weights` - Melhor modelo (maior mAP)
- `yolov4-custom_xxxx.weights` - Modelos salvos periodicamente

## Requisitos do Sistema

- Darknet v5 "Moonlit" (instalado)
- CUDA 13.0 (instalado)
- cuDNN 9.13.0 (instalado)
- OpenCV 4.6.0 (instalado)
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU

## Dicas de Uso

1. **Tempo de treinamento**: Com a RTX 4050, o treinamento leva ~4-5 horas (3000 iterações)
2. **Monitoramento**: Acompanhe a loss - deve convergir gradualmente
3. **Early stopping**: Se a loss parar de diminuir, você pode interromper o treinamento
4. **Inferência**: Use threshold entre 0.3-0.5 para detecção (ajuste conforme necessário)

## Solução de Problemas

### Erro de memória GPU:
- Aumente `subdivisions` no arquivo .cfg (ex: 64 ou 128)
- Reduza resolução para 416x416 se necessário
- Diminua `batch` se necessário

### Loss não converge:
- Verifique se as anotações estão corretas
- Use learning rate conservador (0.0001)
- Evite learning rates altos (>0.001) que causam instabilidade
- Verifique se as imagens e anotações correspondem

### mAP muito baixo:
- Aumente o número de iterações (mínimo 3000)
- Ajuste os parâmetros de data augmentation
- Verifique a qualidade das anotações
- Use learning rate estável (0.0001)
- Certifique-se que o dataset está balanceado

## 📂 Controle de Versão (Git)

### Arquivos NO repositório:
- ✅ Configurações (.cfg, .data, .names)
- ✅ Scripts (setup.sh, train.sh, test.sh)
- ✅ Documentação (.md)
- ✅ Listas de imagens (train.txt, valid.txt)
- ✅ **Estrutura otimizada**: Labels e imagens na mesma pasta

### Arquivos IGNORADOS (.gitignore):
- ❌ `yolov4.conv.137` (162MB - muito grande)
- ❌ `backup/*.weights` (modelos treinados)
- ❌ `*.png` (gráficos de treinamento)
- ❌ Dataset completo (também muito grande)

### Para colaboradores:
1. Clone o repositório
2. Execute `./setup.sh` (baixa os pesos automaticamente)
3. Pronto para usar!

### Por que essa estrutura?
- ⚡ **Clone rápido** (sem arquivos gigantes)
- 🔄 **Reprodutível** (qualquer um pode configurar)
- 📦 **GitHub-friendly** (sem arquivos > 100MB)
- 👥 **Colaborativo** (fácil para outros desenvolvedores)

## 📋 Fluxo de Trabalho Recomendado

```bash
# 1. Primeira vez (setup)
git clone <seu-repo>
cd seu-repo
./setup.sh

# 2. Treinar modelo (Python recomendado)
python3 train_yolo.py

# Ou usar script bash
./train.sh

# 3. Testar resultado
./test.sh

# 4. Commit apenas configurações (não os .weights!)
git add *.cfg *.md *.sh *.py
git commit -m "Configuração final: mAP 18.9% com 3000 iterações"
git push
```

## 🎯 Resultados Obtidos

### **Resultado Final do Treinamento:**
- **mAP@0.50**: 18.90% (resultado sólido para dataset desafiador)
- **Recall**: 24% (encontra 1 em cada 4 objetos)
- **Precision**: 33% (1 em cada 3 detecções corretas)
- **Detecções Corretas**: 106 no conjunto de validação

### **Performance por Classe:**
- **Lesão no quarto traseiro**: 20.66% AP
- **Perda no quarto dianteiro**: 13.23% AP
- **Perda no quarto traseiro**: 22.79% AP (melhor classe)

### **Configuração Final Otimizada:**
- **3000 iterações** (4-5 horas na RTX 4050)
- **Learning rate**: 0.0001 (estável)
- **Resolução**: 512x512 (economia GPU)
- **Subdivisions**: 32 (balanceado)
- **3 classes balanceadas** (removida classe problemática)

## 📖 Documentação Adicional

- **README_YOLOV4_CONFIG.md**: Explicação detalhada do arquivo .cfg
- **obj.names**: Classes do dataset
- **Comentários no código**: Scripts documentados

## Contato

Para dúvidas sobre este projeto de detecção de lesões em carcaças bovinas:
- **Autores**: Felipe e José Pires
- **Projeto**: TCC - Detecção de carcaças bovinas utilizando visão computacional
- **Tecnologias**: Darknet, YOLO v4, CUDA