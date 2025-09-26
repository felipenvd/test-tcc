# Script Python para Treinamento YOLOv4 Inteligente

O script `train_yolo.py` oferece **monitoramento avançado** e **controle inteligente** do treinamento YOLOv4, indo muito além do script bash básico.

## 🚀 Funcionalidades Avançadas

### **📊 Monitoramento em Tempo Real**
- **Gráficos dinâmicos** de loss e mAP
- **Métricas detalhadas** salvas automaticamente
- **Progresso visual** atualizado a cada 50 iterações

### **🧠 Early Stopping Inteligente**
- **Para automaticamente** quando loss estabiliza
- **Evita overfitting** parando na hora certa
- **Patience configurável** (padrão: 200 iterações)

### **📈 Análise Completa**
- **Relatório JSON** com todas as métricas
- **Gráficos finais** em alta resolução
- **Estimativa de tempo** baseada em performance real

### **🛡️ Controle Robusto**
- **Ctrl+C gracioso** (não corrompe modelos)
- **Verificação automática** de arquivos
- **Backup inteligente** dos melhores pesos

## 📦 Instalação

### **1. Instalar dependências:**
```bash
pip install -r requirements.txt
# ou
pip install matplotlib numpy
```

### **2. Verificar instalação:**
```bash
python3 -c "import matplotlib, numpy; print('✅ Dependências OK')"
```

## 🎯 Como Usar

### **Uso Básico (Recomendado):**
```bash
python3 train_yolo.py
```

### **Uso Avançado:**
```bash
# Com configuração personalizada
python3 train_yolo.py --config yolov4-custom.cfg --patience 300

# Com arquivos diferentes
python3 train_yolo.py --data obj.data --weights yolov4.conv.137
```

### **Parâmetros Disponíveis:**
```bash
--config    # Arquivo .cfg (padrão: yolov4-custom.cfg)
--data      # Arquivo .data (padrão: obj.data)
--weights   # Pesos pré-treinados (padrão: yolov4.conv.137)
--patience  # Early stopping (padrão: 200 iterações)
```

## 📊 O que o Script Faz

### **🔍 Verificações Automáticas:**
```
🔍 Verificando arquivos necessários...
✅ yolov4-custom.cfg
✅ obj.data
✅ yolov4.conv.137
✅ obj.names
✅ train.txt
✅ valid.txt
✅ Darknet encontrado e funcionando
```

### **📈 Informações do Dataset:**
```
📊 Dataset Information:
   - Treinamento: 720 imagens
   - Validação: 90 imagens
   - Total: 810 imagens

⚙️ Configuração de Treinamento:
   - Batch size: 64
   - Subdivisions: 64
   - Max iterations: 3000
   - Learning rate: 0.0001 (otimizado para estabilidade)

⏱️ Tempo estimado: 4h 40m
   (baseado em 8.4s por iteração)
```

### **🚀 Monitoramento em Tempo Real:**
```
🚀 Iniciando Treinamento YOLOv4 - Carcaças Bovinas
============================================================
45: loss=2038.979, avg loss=2087.802 [📊 Gráfico atualizado]
46: loss=2026.638, avg loss=2081.685
47: loss=2013.886, avg loss=2074.905
...
1000: loss=234.567, avg loss=245.123 [📈 mAP calculado: 0.7834]
...
```

## 🛑 Early Stopping Inteligente

### **Como Funciona:**
1. **Monitora** loss e mAP continuamente
2. **Conta** iterações sem melhora
3. **Para automaticamente** após 200 iterações estagnadas
4. **Salva** automaticamente o melhor modelo

### **Exemplo de Early Stopping:**
```
1450: loss=156.234, avg loss=158.901
1451: loss=157.123, avg loss=158.967
1452: loss=156.890, avg loss=158.934
...
🛑 Early stopping na iteração 1650
   Sem melhora por 200 iterações
   Melhor loss: 155.234 (iteração 1450)
   Melhor mAP: 0.8234 (iteração 1400)
```

## 📁 Arquivos Gerados

### **Durante o Treinamento:**
- `training_progress.png` - Gráfico atualizado em tempo real

### **Ao Final:**
- `training_report.json` - Métricas completas em JSON
- `training_final.png` - Gráficos finais em alta qualidade
- `backup/yolov4-custom_best.weights` - Melhor modelo

### **Exemplo do JSON:**
```json
{
  "timestamp": "2025-01-15T14:30:45",
  "total_iterations": 1650,
  "elapsed_time_hours": 3.85,
  "initial_loss": 2087.8,
  "final_loss": 155.2,
  "best_loss": 155.2,
  "best_map": 0.8234,
  "iterations": [1, 2, 3, ...],
  "losses": [2087.8, 2045.3, ...],
  "avg_losses": [2087.8, 2066.5, ...],
  "maps": [0.1234, 0.2345, ...]
}
```

## 📊 Gráficos Gerados

### **training_progress.png** (Tempo Real):
- **Loss por iteração** (linha azul)
- **Average Loss** (linha vermelha)
- **mAP** (linha verde, se disponível)

### **training_final.png** (Alta Qualidade):
- **4 subplots** com análise completa:
  1. Loss individual por iteração
  2. Average Loss suavizada
  3. mAP evolution (se calculado)
  4. Tempo por iteração

## 🆚 Comparação: Python vs Bash

| Aspecto | train.sh | train_yolo.py |
|---------|----------|---------------|
| **Monitoramento** | Básico | Avançado ✅ |
| **Early Stopping** | Manual | Automático ✅ |
| **Gráficos** | Nenhum | Tempo Real ✅ |
| **Relatórios** | Nenhum | JSON + PNG ✅ |
| **Controle de Erro** | Simples | Robusto ✅ |
| **Estimativa Tempo** | Nenhuma | Precisa ✅ |
| **Facilidade** | Simples | Intuitivo ✅ |

## 🎯 Quando Usar Cada Um

### **Use `train.sh` quando:**
- ✅ Treinamento rápido e simples
- ✅ Não precisa de monitoramento
- ✅ Quer controle manual total

### **Use `train_yolo.py` quando:**
- ✅ **Quer otimizar tempo** (early stopping)
- ✅ **Precisa de relatórios** para TCC
- ✅ **Quer gráficos bonitos** para apresentação
- ✅ **Treinamento longo** (>2 horas)
- ✅ **Análise científica** detalhada

## 🛠️ Troubleshooting

### **Erro: matplotlib not found**
```bash
pip install matplotlib
# ou
pip install -r requirements.txt
```

### **Erro: Permission denied**
```bash
chmod +x train_yolo.py
```

### **Erro: Darknet not found**
```bash
which darknet
# Deve retornar: /usr/bin/darknet
```

### **Erro: Early stopping muito cedo**
```bash
# Aumentar patience
python3 train_yolo.py --patience 400
```

## 📈 Exemplo de Uso Completo

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Verificar setup
./setup.sh

# 3. Treinar com Python (RECOMENDADO)
python3 train_yolo.py

# 4. Monitorar progresso
# - Observe terminal para métricas
# - Abra training_progress.png para gráficos
# - Ctrl+C para parar graciosamente

# 5. Analisar resultados
# - training_report.json (dados)
# - training_final.png (gráficos)
# - backup/yolov4-custom_best.weights (modelo)
```

## 🎓 Para seu TCC

### **Vantagens Acadêmicas:**
1. **📊 Métricas Precisas**: JSON com todos os dados
2. **📈 Visualizações**: Gráficos profissionais
3. **⚡ Otimização**: Early stopping economiza tempo
4. **📝 Reprodutibilidade**: Configuração documentada
5. **🔬 Análise Científica**: Dados para discussão

### **Apresentação:**
- Use `training_final.png` nos slides
- Cite métricas do `training_report.json`
- Explique early stopping como otimização

## 🏆 Resultado Final

Com o **script Python**, seu treinamento será:
- ⚡ **Mais rápido** (para quando otimizado)
- 📊 **Mais informativo** (métricas detalhadas)
- 🎨 **Mais visual** (gráficos em tempo real)
- 🔬 **Mais científico** (dados para análise)
- 🎯 **Mais profissional** (para seu TCC)

**Autores**: Felipe e José Pires | TCC 2025
**Hardware**: NVIDIA RTX 4050 + Darknet v5 "Moonlit"