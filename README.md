
# CUDA Benchmark: C++ vs. Python

Este é um benchmark básico que compara a utilização da GPU entre CUDA em C++ e Python. O objetivo é testar e medir a diferença de desempenho ao realizar operações paralelas usando uma NVIDIA GeForce RTX 2070.

## Estrutura do Benchmark

O benchmark realiza operações de adição de vetores em uma GPU utilizando CUDA, tanto em C++ quanto em Python. O código foi executado em um ambiente controlado para medir o tempo de execução em diferentes tamanhos de vetor.

## Requisitos

- Uma GPU compatível com CUDA (NVIDIA GeForce RTX 2070 utilizada neste benchmark)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) instalado
- [Microsoft Visual Studio](https://visualstudio.microsoft.com/) para compilar o código C++
- [Visual Studio Code](https://code.visualstudio.com/) ou qualquer outro IDE/editor de sua escolha para executar o código Python

## Preparação do Ambiente

### Código C++

1. **Renomear o Arquivo**: O arquivo de código C++ deve ter a extensão `.cu`.
2. **Compilação**: Use o seguinte comando para compilar o código:
   ```sh
   nvcc -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe" -allow-unsupported-compiler -o cpp-cuda cpp-cuda.cu
   ```
3. **Execução**: Execute o binário gerado:
   ```sh
   .\cpp-cuda.exe
   ```

### Código Python

1. **Configuração do Ambiente**: Instale as bibliotecas necessárias (NumPy e Numba). Você pode fazer isso usando pip:
   ```sh
   pip install numpy numba
   ```
2. **Execução**: Utilize o Visual Studio Code ou qualquer outro IDE/editor para rodar o código Python.

## Resultados do Benchmark

### 1.000.000 Elementos na matriz

**C++:**
1. 0.0099083 segundos
2. 0.0073619 segundos
3. 0.0101045 segundos
4. 0.0080555 segundos
5. 0.0073942 segundos

**Média (C++)**: 0.00856488 segundos

**Python:**
1. 0.2853059 segundos
2. 0.2853059 segundos
3. 0.1937406 segundos
4. 0.1888995 segundos
5. 0.1877501 segundos

**Média (Python)**: 0.2282004 segundos

**Razão (C++/Python)**: C++ ~26.64 vezes mais rápido

### 10.000.000 Elementos na matriz

**C++:**
1. 0.0914478 segundos
2. 0.0755039 segundos
3. 0.0714669 segundos
4. 0.0745260 segundos
5. 0.0691779 segundos

**Média (C++)**: 0.0768245 segundos

**Python:**
1. 0.1841082 segundos
2. 0.1841082 segundos
3. 0.1836843 segundos
4. 0.1810455 segundos
5. 0.1806254 segundos

**Média (Python)**: 0.1827143 segundos

**Razão (C++/Python)**: C++ ~2.38 vezes mais rápido

## Observações

- **GPU Compatível**: Certifique-se de que sua GPU é compatível com CUDA e que você instalou o CUDA Toolkit corretamente.
- **Ambiente de Desenvolvimento**: Utilize o Microsoft Visual Studio para compilar e rodar o código C++ e Visual Studio Code para executar o código Python.
- **Desempenho**: O código em C++ utilizando CUDA apresentou um desempenho significativamente melhor em comparação ao código Python, especialmente com vetores de maior tamanho.
