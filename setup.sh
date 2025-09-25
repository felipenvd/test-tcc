#!/bin/bash

# Script de configuração inicial para o projeto TCC
# Detecção de lesões e perdas em carcaças bovinas com YOLO
# Autores: Felipe e José Pires

echo "======================================================="
echo "  Setup do Projeto - Detecção de Lesões em Carcaças"
echo "======================================================="
echo ""

# Verificar se já existe o arquivo de pesos
if [ -f "yolov4.conv.137" ]; then
    echo "✓ Arquivo yolov4.conv.137 já existe (162MB)"
    echo "  Pulando download..."
else
    echo "📥 Baixando pesos pré-treinados YOLOv4..."
    echo "   Arquivo: yolov4.conv.137 (162MB)"
    echo "   Fonte: GitHub oficial AlexeyAB/darknet"
    echo ""

    # Download dos pesos pré-treinados
    wget -c https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Download concluído com sucesso!"
    else
        echo ""
        echo "❌ Erro no download. Tente novamente ou baixe manualmente:"
        echo "   https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137"
        exit 1
    fi
fi

echo ""

# Verificar se o diretório backup existe
if [ ! -d "backup" ]; then
    echo "📁 Criando diretório backup/..."
    mkdir -p backup
    echo "✓ Diretório backup/ criado"
else
    echo "✓ Diretório backup/ já existe"
fi

echo ""

# Verificar arquivos essenciais
echo "🔍 Verificando arquivos essenciais..."

files_to_check=("obj.names" "obj.data" "yolov4-custom.cfg" "train.txt" "valid.txt")
all_ok=true

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "❌ $file - ARQUIVO AUSENTE!"
        all_ok=false
    fi
done

echo ""

if [ "$all_ok" = true ]; then
    echo "🎉 Setup concluído com sucesso!"
    echo ""
    echo "📋 Próximos passos:"
    echo "   1. Para treinar: ./train.sh"
    echo "   2. Para testar:  ./test.sh"
    echo "   3. Documentação: README_YOLO.md"
    echo ""
    echo "💾 Estrutura do dataset:"
    echo "   - Treinamento: $(cat train.txt | wc -l) imagens"
    echo "   - Validação:   $(cat valid.txt | wc -l) imagens"
    echo "   - Classes:     4 (lesões e perdas nos quartos)"
else
    echo "⚠️  Alguns arquivos estão ausentes. Verifique a configuração."
    exit 1
fi

echo ""
echo "📝 IMPORTANTE:"
echo "   - O arquivo yolov4.conv.137 NÃO deve ser commitado no Git"
echo "   - Já foi adicionado ao .gitignore automaticamente"
echo "   - Para colaboradores: executar ./setup.sh antes do primeiro uso"
echo ""
echo "======================================================="