#!/bin/bash

# Script para treinar modelo YOLO para detecção de lesões em carcaças bovinas
# Autor: Felipe e José Pires

echo "=== Iniciando Treinamento do Modelo YOLO ==="
echo "Dataset: Detecção de lesões e perdas em carcaças bovinas"
echo "Classes: 4 (lesao_superficial, lesao_profunda, perda_corte, perda_abscesso)"
echo "Imagens de treinamento: 720"
echo "Imagens de validação: 90"
echo ""

# Verificar se os arquivos necessários existem
echo "Verificando arquivos necessários..."

if [ ! -f "obj.data" ]; then
    echo "ERRO: Arquivo obj.data não encontrado!"
    exit 1
fi

if [ ! -f "yolov4-custom.cfg" ]; then
    echo "ERRO: Arquivo yolov4-custom.cfg não encontrado!"
    exit 1
fi

if [ ! -f "yolov4.conv.137" ]; then
    echo "ERRO: Arquivo yolov4.conv.137 não encontrado!"
    exit 1
fi

if [ ! -f "obj.names" ]; then
    echo "ERRO: Arquivo obj.names não encontrado!"
    exit 1
fi

echo "✓ Todos os arquivos necessários encontrados"
echo ""

# Criar diretório de backup se não existir
mkdir -p backup

echo "Iniciando treinamento com Darknet..."
echo "Comando: darknet detector train obj.data yolov4-custom.cfg yolov4.conv.137 -map -clear"
echo ""

# Executar o treinamento
darknet detector train obj.data yolov4-custom.cfg yolov4.conv.137 -map -clear

echo ""
echo "=== Treinamento Concluído ==="
echo "Os pesos do modelo treinado estão salvos em: backup/"