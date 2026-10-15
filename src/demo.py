"""
demo.py - treina a rede do network.py (MNIST) e classifica imagens suas.

Requisitos: numpy, pillow  (pip install numpy pillow)
Aceita qualquer formato que o Pillow leia (PNG, JPG, BMP, PGM...).

Se a imagem já for 28x28, ela é usada como está (só converte para cinza e
inverte se o fundo for claro). Se for de outro tamanho, o script recorta o
traço, redimensiona para 20x20 e centraliza em 28x28, como no MNIST.

Uso (use python3):
  python3 demo.py treinar --epochs 30
  python3 demo.py treinar --hidden 64 32 --saida pesos_64_32.npz
  python3 demo.py prever meu_numero.png
  python3 demo.py prever meu_numero.png --pesos pesos_64_32.npz
  python3 demo.py prever meu_numero.png --sem-treino --hidden 64 32
  python3 demo.py amostra 5 --pesos pesos_64_32.npz
"""
import argparse
import os
import random

import numpy as np
from PIL import Image

import network
import mnist_loader

ARQUIVO_PESOS = "pesos.npz"


# ---------- salvar / carregar pesos ----------
def salvar(net, caminho=ARQUIVO_PESOS):
    dados = {"sizes": np.array(net.sizes)}
    for i, (b, w) in enumerate(zip(net.biases, net.weights)):
        dados[f"b{i}"] = b
        dados[f"w{i}"] = w
    np.savez(caminho, **dados)
    print(f"Pesos salvos em {caminho}")


def carregar(caminho=ARQUIVO_PESOS):
    if not os.path.exists(caminho):
        raise SystemExit(f"Arquivo '{caminho}' não existe ainda.\n"
                         "Rode primeiro: python3 demo.py treinar\n"
                         "(ou use --sem-treino para testar com pesos aleatórios)")
    dados = np.load(caminho)
    sizes = [int(s) for s in dados["sizes"]]
    net = network.Network(sizes)
    n = len(sizes) - 1
    net.biases = [dados[f"b{i}"] for i in range(n)]
    net.weights = [dados[f"w{i}"] for i in range(n)]
    print(f"Pesos carregados de {caminho} (arquitetura {sizes})")
    return net


# ---------- imagem -> vetor 784x1 ----------
def imagem_para_vetor(caminho, salvar_debug="entrada_processada.png"):
    img = Image.open(caminho)

    # PNG com transparência: compõe sobre fundo branco
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        fundo = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(fundo, img)
    img = img.convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # MNIST = dígito claro em fundo escuro. Se as bordas forem claras, inverte.
    bordas = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if bordas.mean() > 0.5:
        arr = 1.0 - arr

    if arr.shape == (28, 28):
        final = arr  # já no formato certo: usa como está
    else:
        mascara = arr > 0.2
        if not mascara.any():
            raise SystemExit("Não encontrei nenhum traço na imagem.")
        ys, xs = np.where(mascara)
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        h, w = arr.shape
        escala = 20.0 / max(h, w)
        novo_w, novo_h = max(1, round(w * escala)), max(1, round(h * escala))
        pequeno = Image.fromarray((arr * 255).astype(np.uint8)).resize(
            (novo_w, novo_h), Image.LANCZOS)

        tela = Image.new("L", (28, 28), 0)
        tela.paste(pequeno, ((28 - novo_w) // 2, (28 - novo_h) // 2))
        final = np.asarray(tela, dtype=np.float32) / 255.0

    if salvar_debug:
        Image.fromarray((final * 255).astype(np.uint8)).save(salvar_debug)
    return final.reshape(784, 1)


def salvar_png(vetor, caminho):
    Image.fromarray((vetor.reshape(28, 28) * 255).astype(np.uint8)).save(caminho)


def mostrar_ascii(vetor):
    """Desenha a imagem 28x28 no terminal."""
    chars = " .:-=+*#%@"
    print("\nEntrada da rede:")
    for linha in vetor.reshape(28, 28):
        print("".join(chars[min(int(v * 10), 9)] * 2 for v in linha))


def mostrar_previsao(net, x):
    saida = net.feedforward(x)
    palpite = int(np.argmax(saida))
    print(f"\nPalpite da rede: {palpite}\n")
    for d in range(10):
        print(f"  {d}: {saida[d, 0]:.3f} {'#' * int(saida[d, 0] * 30)}")
    return palpite


# ---------- comandos ----------
def cmd_treinar(args):
    treino, validacao, teste = mnist_loader.load_data_wrapper()
    treino, teste = list(treino), list(teste)
    net = network.Network([784] + args.hidden + [10])
    print(f"Arquitetura: {net.sizes}")
    net.SGD(treino, args.epochs, args.batch, args.eta, test_data=teste)
    salvar(net, args.saida)


def cmd_prever(args):
    if args.sem_treino:
        net = network.Network([784] + args.hidden + [10])
        print("(Rede SEM treino, pesos aleatórios)")
    else:
        net = carregar(args.pesos)
    x = imagem_para_vetor(args.imagem)
    # mostrar_ascii(x)
    mostrar_previsao(net, x)
    print("\n(imagem processada salva em entrada_processada.png)")


def cmd_amostra(args):
    _, _, teste = mnist_loader.load_data_wrapper()
    teste = list(teste)
    net = (network.Network([784] + args.hidden + [10])
           if args.sem_treino else carregar(args.pesos))
    for i, (x, y) in enumerate(random.sample(teste, args.n)):
        salvar_png(x, f"amostra_{i}.png")
        print(f"\n=== amostra_{i}.png (rótulo real: {y}) ===")
        # mostrar_ascii(x)
        mostrar_previsao(net, x)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("treinar")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=10)
    t.add_argument("--eta", type=float, default=3.0)
    t.add_argument("--hidden", type=int, nargs="+", default=[30])
    t.add_argument("--saida", default=ARQUIVO_PESOS)
    t.set_defaults(f=cmd_treinar)

    pr = sub.add_parser("prever")
    pr.add_argument("imagem")
    pr.add_argument("--sem-treino", action="store_true")
    pr.add_argument("--hidden", type=int, nargs="+", default=[30])
    pr.add_argument("--pesos", default=ARQUIVO_PESOS)
    pr.set_defaults(f=cmd_prever)

    a = sub.add_parser("amostra")
    a.add_argument("n", type=int, nargs="?", default=3)
    a.add_argument("--sem-treino", action="store_true")
    a.add_argument("--hidden", type=int, nargs="+", default=[30])
    a.add_argument("--pesos", default=ARQUIVO_PESOS)
    a.set_defaults(f=cmd_amostra)

    args = p.parse_args()
    args.f(args)