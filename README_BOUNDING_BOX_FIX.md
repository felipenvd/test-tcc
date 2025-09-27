# 🔧 Correção de Bounding Boxes: Darknet Coordinate System

**Problema Comum**: Bounding boxes desalinhados ao integrar detecções do Darknet em código Python

## 📋 Resumo do Problema

Ao usar o Darknet YOLO para detecção de objetos e integrar os resultados em código Python (OpenCV/matplotlib), os bounding boxes apareciam **deslocados e menores** que o esperado, mesmo com as coordenadas corretas sendo retornadas pelo Darknet.

### 🔍 Sintomas
- ✅ Darknet `test.sh` mostra detecções corretas
- ❌ Implementação Python mostra boxes pequenos e deslocados
- ❌ Boxes aparecem no canto superior esquerdo da imagem
- ❌ Tamanhos muito menores que o esperado

## 🎯 Root Cause Identificado

**INTERPRETAÇÃO INCORRETA DO SISTEMA DE COORDENADAS DO DARKNET**

### ❌ Interpretação Errada (Como fizemos inicialmente)
```python
# STDOUT do Darknet: x=1116 y=881 w=39 h=52
x_center = 1116  # ❌ ERRADO: Interpretando x como centro
y_center = 881   # ❌ ERRADO: Interpretando y como centro
width = 39
height = 52

# Conversão incorreta
x1 = int(x_center - width/2)   # 1116 - 19 = 1097
y1 = int(y_center - height/2)  # 881 - 26 = 855
x2 = int(x_center + width/2)   # 1116 + 19 = 1135
y2 = int(y_center + height/2)  # 881 + 26 = 907

# Resultado: Box pequeno (38x52) em posição errada
```

### ✅ Interpretação Correta
```python
# STDOUT do Darknet: x=1116 y=881 w=39 h=52
x_top_left = 1116  # ✅ CORRETO: x é canto superior esquerdo
y_top_left = 881   # ✅ CORRETO: y é canto superior esquerdo
width = 39
height = 52

# Conversão correta
x1 = int(x_top_left)           # 1116
y1 = int(y_top_left)           # 881
x2 = int(x_top_left + width)   # 1116 + 39 = 1155
y2 = int(y_top_left + height)  # 881 + 52 = 933

# Resultado: Box correto (39x52) na posição certa
```

## 📖 Diferença entre Formatos

### 🔄 Formato de Anotação YOLO (Arquivos .txt)
```
# YOLO annotation format (normalizado 0-1, centro)
class_id x_center_norm y_center_norm width_norm height_norm
0 0.291 0.408 0.010 0.024
```
- **x,y = CENTRO** normalizado (0-1)
- Usado em arquivos de anotação `.txt`

### 📺 Formato de Saída Darknet (Console/stdout)
```
# Darknet detector test output (pixels absolutos, canto)
Lesao no quarto traseiro	c=27.855274%	x=1116	y=881	w=39	h=52
```
- **x,y = CANTO SUPERIOR ESQUERDO** em pixels absolutos
- Usado na saída do console do comando `detector test`

## 🛠️ Solução Implementada

### Função Corrigida (Python)
```python
def parse_darknet_detection(stdout_line):
    """Parse correto da saída do Darknet detector test"""
    parts = line.split('\t')

    # Parse das coordenadas
    x_top_left = float(x_part.split('=')[1])    # Canto superior esquerdo
    y_top_left = float(y_part.split('=')[1])    # Canto superior esquerdo
    width = float(w_part.split('=')[1])         # Largura
    height = float(h_part.split('=')[1])        # Altura

    # Converter para formato OpenCV (x1,y1,x2,y2)
    x1 = int(x_top_left)
    y1 = int(y_top_left)
    x2 = int(x_top_left + width)
    y2 = int(y_top_left + height)

    return [x1, y1, x2, y2]
```

## 🧪 Como Validar a Correção

### 1. Gerar Imagem de Referência com Darknet
```bash
# Gera predictions.jpg com bounding boxes corretos
darknet detector test obj.data yolov4-custom.cfg weights.weights image.jpg -dont_show
```

### 2. Comparar com Implementação Python
```python
# Sua implementação deve gerar boxes na MESMA posição
boxes = parse_darknet_detection(image_path)
draw_boxes(image, boxes)
# Compare visualmente com predictions.jpg
```

### 3. Teste de Múltiplas Interpretações
```python
# Teste diferentes interpretações para verificar qual é correta
interpretations = [
    ("Centro", [x_center-w/2, y_center-h/2, x_center+w/2, y_center+h/2]),
    ("Canto Sup. Esq.", [x, y, x+w, y+h]),  # <- Esta deve ser a correta
    ("Y Invertido", [x-w/2, img_h-y-h/2, x+w/2, img_h-y+h/2])
]
```

## 📊 Resultados da Correção

### Antes da Correção
- ❌ Boxes pequenos (39x52 pixels)
- ❌ Posição: canto superior esquerdo da imagem
- ❌ Não alinhados com objetos detectados

### Depois da Correção
- ✅ Boxes no tamanho correto (39x52 pixels)
- ✅ Posição: sobre os objetos detectados
- ✅ Perfeitamente alinhados com `predictions.jpg`

## 🚨 Pegadinha Comum

**Por que isso acontece?**

1. **Documentação Ambígua**: Muitas fontes falam sobre "coordenadas do centro" referindo-se ao formato YOLO de anotação
2. **Formatos Misturados**: Darknet usa formatos diferentes para entrada (anotações) vs saída (detecções)
3. **Convenções Diferentes**: OpenCV usa (x1,y1,x2,y2) enquanto YOLO usa (center_x, center_y, width, height)

## 🔧 Implementação Completa

Ver arquivo `analise_modelo.ipynb` células 4-5 para implementação completa com:
- ✅ Parsing correto do stdout do Darknet
- ✅ Método alternativo com `save_labels`
- ✅ Validação automática de coordenadas
- ✅ Sistema de debug para identificar problemas

## 📚 Referências

- [Darknet YOLO Documentation](https://github.com/pjreddie/darknet)
- [YOLO Coordinate System Explanation](https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection)
- [OpenCV Rectangle Drawing](https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9)

## 👥 Créditos

**Problema Identificado e Resolvido Por**: Felipe e José Pires
**Projeto**: TCC - Detecção de Lesões em Carcaças Bovinas com YOLO v4
**Data**: Setembro 2025

---

💡 **Dica**: Se você está enfrentando problemas similares com bounding boxes desalinhados, verifique primeiro se não está interpretando incorretamente o sistema de coordenadas da sua ferramenta de detecção!