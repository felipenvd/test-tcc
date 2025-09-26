#!/bin/bash

# Script para testar modelo YOLO treinado
# Autor: Felipe e José Pires

echo "=== Testando Modelo YOLO Treinado ==="
echo ""

# Verificar se o modelo treinado existe
WEIGHTS_FILE="backup/yolov4-custom_best.weights"

if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "ERRO: Modelo treinado não encontrado em $WEIGHTS_FILE"
    echo "Execute primeiro o treinamento com ./train.sh"
    exit 1
fi

echo "✓ Modelo treinado encontrado: $WEIGHTS_FILE"
echo ""

# Testar com uma imagem específica (se fornecida como argumento)
if [ $# -eq 1 ]; then
    IMAGE_PATH="$1"
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "ERRO: Imagem não encontrada: $IMAGE_PATH"
        exit 1
    fi

    echo "Testando com imagem: $IMAGE_PATH"
    darknet detector test obj.data yolov4-custom.cfg "$WEIGHTS_FILE" "$IMAGE_PATH" -thresh 0.25
else
    echo "Calculando mAP (Mean Average Precision) no conjunto de validação..."
    darknet detector map obj.data yolov4-custom.cfg "$WEIGHTS_FILE"
fi

echo ""
echo "=== Teste Concluído ==="