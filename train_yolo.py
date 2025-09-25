#!/usr/bin/env python3
"""
Script Python para Treinamento YOLOv4 - Detecção de Lesões em Carcaças Bovinas
Autores: Felipe e José Pires | TCC 2025
Hardware: NVIDIA RTX 4050 Laptop GPU + Darknet v5 "Moonlit"

Funcionalidades:
- Monitoramento automático de loss e mAP
- Early stopping inteligente
- Gráficos de progresso em tempo real
- Backup automático dos melhores modelos
- Relatório final de treinamento
"""

import os
import sys
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime, timedelta
import json
import signal
import argparse

class YOLOTrainer:
    def __init__(self, config_file="yolov4-custom.cfg", data_file="obj.data", weights="yolov4.conv.137"):
        self.config_file = config_file
        self.data_file = data_file
        self.weights = weights
        self.backup_dir = "backup"

        # Listas para armazenar métricas
        self.iterations = []
        self.losses = []
        self.avg_losses = []
        self.maps = []
        self.times = []

        # Configurações de early stopping
        self.best_map = 0.0
        self.best_loss = float('inf')
        self.patience = 200  # iterações sem melhora
        self.no_improve_count = 0
        self.start_time = None

        # Controle de processo
        self.process = None
        self.training_stopped = False

    def signal_handler(self, signum, frame):
        """Manipula Ctrl+C graciosamente"""
        print(f"\n🛑 Sinal {signum} recebido. Parando treinamento graciosamente...")
        self.training_stopped = True
        if self.process:
            self.process.terminate()
        self.generate_report()
        sys.exit(0)

    def check_files(self):
        """Verifica se todos os arquivos necessários existem"""
        print("🔍 Verificando arquivos necessários...")

        required_files = [
            self.config_file,
            self.data_file,
            self.weights,
            "obj.names",
            "train.txt",
            "valid.txt"
        ]

        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)

        if missing_files:
            print("❌ Arquivos ausentes:")
            for file in missing_files:
                print(f"   - {file}")
            return False

        # Verificar se darknet está disponível
        try:
            result = subprocess.run(["darknet"], capture_output=True, text=True, timeout=5)
            print("✅ Darknet encontrado e funcionando")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Darknet não encontrado ou não funciona")
            return False

        print("✅ Todos os arquivos necessários encontrados")
        return True

    def get_dataset_info(self):
        """Obtém informações sobre o dataset"""
        try:
            with open("train.txt", "r") as f:
                train_count = len(f.readlines())
            with open("valid.txt", "r") as f:
                valid_count = len(f.readlines())

            print(f"📊 Dataset Information:")
            print(f"   - Treinamento: {train_count} imagens")
            print(f"   - Validação: {valid_count} imagens")
            print(f"   - Total: {train_count + valid_count} imagens")

            return train_count, valid_count
        except Exception as e:
            print(f"⚠️ Erro ao ler informações do dataset: {e}")
            return 0, 0

    def parse_config(self):
        """Extrai informações importantes do arquivo de configuração"""
        try:
            with open(self.config_file, "r") as f:
                content = f.read()

            # Extrair parâmetros principais
            batch = re.search(r'batch=(\d+)', content)
            subdivisions = re.search(r'subdivisions=(\d+)', content)
            max_batches = re.search(r'max_batches\s*=\s*(\d+)', content)
            learning_rate = re.search(r'learning_rate=([\d.]+)', content)

            config_info = {
                'batch': int(batch.group(1)) if batch else 64,
                'subdivisions': int(subdivisions.group(1)) if subdivisions else 16,
                'max_batches': int(max_batches.group(1)) if max_batches else 8000,
                'learning_rate': float(learning_rate.group(1)) if learning_rate else 0.001
            }

            print(f"⚙️ Configuração de Treinamento:")
            print(f"   - Batch size: {config_info['batch']}")
            print(f"   - Subdivisions: {config_info['subdivisions']}")
            print(f"   - Max iterations: {config_info['max_batches']}")
            print(f"   - Learning rate: {config_info['learning_rate']}")

            return config_info
        except Exception as e:
            print(f"⚠️ Erro ao analisar configuração: {e}")
            return {}

    def estimate_training_time(self, config_info, sample_time=10.0):
        """Estima tempo total de treinamento baseado em amostra"""
        max_batches = config_info.get('max_batches', 2000)
        estimated_total = max_batches * sample_time

        hours = estimated_total // 3600
        minutes = (estimated_total % 3600) // 60

        print(f"⏱️ Tempo estimado: {hours:.0f}h {minutes:.0f}m")
        print(f"   (baseado em {sample_time:.1f}s por iteração)")

        return estimated_total

    def parse_training_output(self, line):
        """Extrai métricas da saída do darknet"""
        # Padrão para loss: "45: loss=2038.979, avg loss=2087.802"
        loss_pattern = r'(\d+):\s+loss=([\d.]+),\s+avg\s+loss=([\d.]+)'
        loss_match = re.search(loss_pattern, line)

        if loss_match:
            iteration = int(loss_match.group(1))
            current_loss = float(loss_match.group(2))
            avg_loss = float(loss_match.group(3))

            self.iterations.append(iteration)
            self.losses.append(current_loss)
            self.avg_losses.append(avg_loss)

            # Calcular tempo decorrido
            if self.start_time:
                elapsed = time.time() - self.start_time
                self.times.append(elapsed)

            return iteration, current_loss, avg_loss

        # Padrão para mAP: "mean_average_precision (mAP@0.50) = 0.653245"
        map_pattern = r'mean_average_precision.*?=\s+([\d.]+)'
        map_match = re.search(map_pattern, line)

        if map_match:
            map_value = float(map_match.group(1))
            self.maps.append(map_value)
            return None, None, None, map_value

        return None, None, None, None

    def check_early_stopping(self, current_loss, current_map=None):
        """Verifica condições de early stopping"""
        improved = False

        # Verificar melhora na loss
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            improved = True

        # Verificar melhora no mAP (se disponível)
        if current_map and current_map > self.best_map:
            self.best_map = current_map
            improved = True

        if improved:
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        # Retornar True se deve parar
        return self.no_improve_count >= self.patience

    def update_plots(self, iteration):
        """Atualiza gráficos de progresso em tempo real"""
        if len(self.iterations) < 10:  # Esperar algumas iterações
            return

        plt.clf()

        # Subplot 1: Loss
        plt.subplot(2, 1, 1)
        plt.plot(self.iterations, self.losses, 'b-', alpha=0.6, label='Loss')
        plt.plot(self.iterations, self.avg_losses, 'r-', linewidth=2, label='Avg Loss')
        plt.xlabel('Iterações')
        plt.ylabel('Loss')
        plt.title('Progresso do Treinamento - Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: mAP (se disponível)
        if self.maps:
            plt.subplot(2, 1, 2)
            map_iterations = self.iterations[::100][:len(self.maps)]  # mAP calculado a cada 100 iter
            plt.plot(map_iterations, self.maps, 'g-', linewidth=2, label='mAP@0.5')
            plt.xlabel('Iterações')
            plt.ylabel('mAP')
            plt.title('Progresso do Treinamento - mAP')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.pause(0.1)

    def train(self):
        """Executa o treinamento com monitoramento"""
        print("🚀 Iniciando Treinamento YOLOv4 - Carcaças Bovinas")
        print("=" * 60)

        # Configurar manipulador de sinal
        signal.signal(signal.SIGINT, self.signal_handler)

        # Verificações iniciais
        if not self.check_files():
            print("❌ Verificação de arquivos falhou. Abortando.")
            return False

        train_count, valid_count = self.get_dataset_info()
        config_info = self.parse_config()

        # Criar diretório de backup
        os.makedirs(self.backup_dir, exist_ok=True)

        # Estimar tempo
        sample_time = 8.4  # segundos por iteração (observado na RTX 4050)
        estimated_time = self.estimate_training_time(config_info, sample_time)

        print("\n🎯 Iniciando treinamento...")
        print("   Pressione Ctrl+C para parar graciosamente")
        print("=" * 60)

        # Comando darknet
        cmd = [
            "darknet", "detector", "train",
            self.data_file,
            self.config_file,
            self.weights,
            "-map", "-clear"
        ]

        self.start_time = time.time()

        try:
            # Iniciar processo
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Monitorar saída
            for line in iter(self.process.stdout.readline, ''):
                if self.training_stopped:
                    break

                print(line.strip())

                # Extrair métricas
                result = self.parse_training_output(line)
                if result[0] is not None:  # Se foi uma linha de loss
                    iteration, current_loss, avg_loss = result[:3]
                    current_map = result[3] if len(result) > 3 else None

                    # Verificar early stopping
                    should_stop = self.check_early_stopping(current_loss, current_map)

                    # Atualizar gráficos a cada 50 iterações
                    if iteration % 50 == 0:
                        self.update_plots(iteration)

                    # Early stopping
                    if should_stop:
                        print(f"\n🛑 Early stopping na iteração {iteration}")
                        print(f"   Sem melhora por {self.patience} iterações")
                        self.process.terminate()
                        break

            # Aguardar conclusão
            self.process.wait()

        except Exception as e:
            print(f"❌ Erro durante treinamento: {e}")
            return False

        print("\n✅ Treinamento concluído!")
        self.generate_report()
        return True

    def generate_report(self):
        """Gera relatório final do treinamento"""
        if not self.iterations:
            print("⚠️ Nenhuma métrica coletada para gerar relatório")
            return

        print("\n" + "=" * 60)
        print("📊 RELATÓRIO FINAL DO TREINAMENTO")
        print("=" * 60)

        # Informações básicas
        total_iterations = len(self.iterations)
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        print(f"⏱️ Tempo total: {elapsed_time/3600:.2f} horas")
        print(f"🔢 Total de iterações: {total_iterations}")
        print(f"📈 Loss inicial: {self.losses[0]:.2f}")
        print(f"📉 Loss final: {self.losses[-1]:.2f}")
        print(f"🎯 Melhor loss: {self.best_loss:.2f}")

        if self.maps:
            print(f"📊 Melhor mAP: {self.best_map:.4f}")

        # Salvar métricas em JSON
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_iterations": total_iterations,
            "elapsed_time_hours": elapsed_time/3600,
            "initial_loss": self.losses[0],
            "final_loss": self.losses[-1],
            "best_loss": self.best_loss,
            "best_map": self.best_map,
            "iterations": self.iterations,
            "losses": self.losses,
            "avg_losses": self.avg_losses,
            "maps": self.maps
        }

        with open("training_report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        # Gerar gráfico final
        self.generate_final_plots()

        print("\n📁 Arquivos gerados:")
        print("   - training_report.json (métricas detalhadas)")
        print("   - training_progress.png (gráfico de progresso)")
        print("   - training_final.png (gráfico final)")

    def generate_final_plots(self):
        """Gera gráficos finais do treinamento"""
        plt.figure(figsize=(12, 8))

        # Loss
        plt.subplot(2, 2, 1)
        plt.plot(self.iterations, self.losses, 'b-', alpha=0.6, label='Loss')
        plt.xlabel('Iterações')
        plt.ylabel('Loss')
        plt.title('Loss por Iteração')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(self.iterations, self.avg_losses, 'r-', linewidth=2)
        plt.xlabel('Iterações')
        plt.ylabel('Average Loss')
        plt.title('Average Loss por Iteração')
        plt.grid(True, alpha=0.3)

        # mAP (se disponível)
        if self.maps:
            plt.subplot(2, 2, 3)
            map_iterations = self.iterations[::100][:len(self.maps)]
            plt.plot(map_iterations, self.maps, 'g-', linewidth=2)
            plt.xlabel('Iterações')
            plt.ylabel('mAP@0.5')
            plt.title('mAP por Iteração')
            plt.grid(True, alpha=0.3)

        # Tempo por iteração
        if self.times:
            plt.subplot(2, 2, 4)
            time_per_iter = np.diff(self.times)
            plt.plot(self.iterations[1:], time_per_iter, 'orange', alpha=0.6)
            plt.xlabel('Iterações')
            plt.ylabel('Tempo (s)')
            plt.title('Tempo por Iteração')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_final.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Treinamento YOLOv4 para Carcaças Bovinas')
    parser.add_argument('--config', default='yolov4-custom.cfg', help='Arquivo de configuração')
    parser.add_argument('--data', default='obj.data', help='Arquivo de dados')
    parser.add_argument('--weights', default='yolov4.conv.137', help='Pesos pré-treinados')
    parser.add_argument('--patience', type=int, default=200, help='Early stopping patience')

    args = parser.parse_args()

    # Criar e executar trainer
    trainer = YOLOTrainer(args.config, args.data, args.weights)
    trainer.patience = args.patience

    success = trainer.train()

    if success:
        print("\n🎉 Treinamento concluído com sucesso!")
        print("📝 Verifique os arquivos de relatório gerados.")
    else:
        print("\n❌ Treinamento falhou ou foi interrompido.")

if __name__ == "__main__":
    main()