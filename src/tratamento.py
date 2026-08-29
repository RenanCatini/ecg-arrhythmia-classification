import wfdb
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm
from scipy.signal import filtfilt, iirnotch


def butterworth_passa_alta(              
    sinal, fs=360, freq_corte=0.05, ordem=9
):
    """Aplica filtro passa-faixa Butterworth no sinal de ECG.

    - freq_baixa = 0.5 Hz (remove desvio de linha de base / respiracao)
    - freq_alta = 45.0 Hz (remove ruido muscular e rede eletrica de 60 Hz)
    """
    # 1. Frequência de Nyquist (metade da taxa de amostragem)
    nyquist = 0.5 * fs

    # 2. Frequências de corte normalizadas (devem estar entre 0 e 1)
    corte = freq_corte / nyquist

    # 3. Calcula os coeficientes b e a do filtro Butterworth
    sos = butter(ordem, corte, btype="highpass", output="sos")

    # 4. Aplica o filtro sem defasagem temporal (fase zero)
    sinal_filtrado = sosfiltfilt(sos, sinal)

    return sinal_filtrado

def filtro_notch(sinal, fs=360, freq_rejeicao=60.0, Q=10.0):
    """Aplica um filtro Notch (rejeita-faixa) IIR no sinal.

    - freq_rejeicao: Frequência central a ser eliminada (ex: 60.0 Hz).
    - Q: Fator de qualidade (largura de banda = freq_rejeicao / Q). Quanto maior
    o Q, mais estreita é a faixa rejeitada (evita atenuar o QRS).
    """
    # 1. Projeta os coeficientes b e a do filtro Notch
    b, a = iirnotch(w0=freq_rejeicao, Q=Q, fs=fs)

    # 2. Aplica o filtro bidirecionalmente (fase zero)
    sinal_filtrado = filtfilt(b, a, sinal)

    return sinal_filtrado

def processamento():
    PRE_R = 100
    POST_R = 200

    # Mapa para renomear classes dos batimentos
    mapa_aami = {
        # Normais
        'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',

        # Classe S
        'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',

        # Classe V
        'V': 'V', '!': 'V', 'E': 'V',

        # Classe F
        'F': 'F',

        # Classe Q
        '/': 'Q', 'f': 'Q', 'Q': 'Q'
    }

    REGISTROS = [
        "100", "101", "103", "105", "106", "108", "109", "111", "112", "200", "215",
        "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124", 
        "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214", 
        "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234",
    ] 
    # Registros eliminados: "102", "104", "107", "217"

    # Onde os batimentos tratados ficarão salvos antes de converter para pandas  
    dados_finais = []

    for registro in tqdm(REGISTROS, desc='Processando Exames'):
        # Bloco try apenas temporario
        try:
            # ----------------------- LEITURA DO SINAL -----------------------
            sinal_atual = wfdb.rdrecord(f"../database/raw/{registro}")
            anotacao_atual = wfdb.rdann(f"../database/raw/{registro}", 'atr')

            canais = sinal_atual.sig_name

            # Escolher a derivação MLII
            if 'MLII' in canais:
                indice_canal = canais.index('MLII')
            else:
                print(f'Registro {registro} não possui MLII!')
                indice_canal = 0

            # Pegar o sinal do exame com o canal correto
            ecg_bruto = sinal_atual.p_signal[:,indice_canal]

            # ---- Filtrar ruídos ----
            # Filtro passa-alta para filtrar valores menores que 0.05Hz
            ecg_filtrado_1 = butterworth_passa_alta(ecg_bruto)

            # Filtros rejeita-faixa para retirar filtro de linha de energia de 60Hz e harmônicos
            ecg_filtrado_2 = filtro_notch(ecg_filtrado_1, freq_rejeicao=60.0, Q=20.0)
            ecg_filtrado_3 = filtro_notch(ecg_filtrado_2, freq_rejeicao=120.0, Q=20.0)
            ecg_filtrado_final = filtro_notch(ecg_filtrado_3, freq_rejeicao=180.0, Q=20.0)

            # Picos R e qual sua classe
            indices_picos = anotacao_atual.sample
            simbolos = pd.Series(anotacao_atual.symbol).map(mapa_aami)

            #--------------------- INTERVALOS RR's --------------------
            vetor_rr = np.diff(indices_picos ) / 360
            rr_medio = np.mean(vetor_rr)

            #------------------- EXTRAIR BATIMENTOS -------------------
            total_amostras = len(ecg_filtrado_final)

            # Descarta os primeiros e ultimos batimentos            
            for i in range(1, len(indices_picos)-1):
                pico = indices_picos[i]
                classe = simbolos[i]

                # Se o batimento for nulo, já descarta
                if pd.isna(classe):
                    continue

                # Encontrar intervalo RR anterior e posterior do batimento
                rr_pre = (pico - indices_picos[i-1]) / 360
                rr_pos = (indices_picos[i+1] - pico) / 360
                rr_pre_relativo = rr_pre / rr_medio
                rr_pos_relativo = rr_pos / rr_medio

                # Intervalos do batimento
                inicio = pico - PRE_R
                fim    = pico + POST_R

                # Evitar erros de erro de continuidade
                if inicio < 0 or fim > total_amostras:
                    continue

                # Separar batimento: 100 pontos antes de R, e 200 pontos depois
                segmento = ecg_filtrado_final[inicio:fim]

                # Normalizar por Z-Score
                media_batimento_atual = np.mean(segmento)
                desvio_batimento_atual = np.std(segmento)

                if desvio_batimento_atual != 0:
                    segmento_normalizado = (segmento - media_batimento_atual) / desvio_batimento_atual
                else:
                    segmento_normalizado = segmento - media_batimento_atual

                # Salvar tudo em uma lista com as informações: segmento, id paciente, classe e as informações dos rr's
                linha = [registro] +  list(segmento_normalizado) + [rr_pre, rr_pos, rr_pre_relativo, rr_pos_relativo, classe]
                dados_finais.append(linha)

        except FileNotFoundError:
            tqdm.write(f"Arquivo {registro} não existe!")
        except Exception as e:
            tqdm.write(f"Erro ao ler {registro}: {e}")

    # ---------------- Salvar informações dos batimentos ----------------
    colunas_pontos = [f'ponto_{pontos}' for pontos in range(0,300)]
    colunas_infos = ["rr_pre", "rr_pos", "rr_pre_relativo", "rr_pos_relativo", "padrao_aami"]
    todas_colunas = ['registro'] + colunas_pontos + colunas_infos

    # Criar df
    df = pd.DataFrame(dados_finais, columns=todas_colunas)
    
    return df
    #-------------------------------------------------------------------


if __name__ == "__main__":
    print("Executando processamento...")
    ecgs = processamento()

    # Separ o conjunto em teste e treino seguindo a separação do conjunto
    pacientes_treino = [
        "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
        "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
    ]

    pacientes_teste = [
        "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
        "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
    ]

    ecgs['tipo'] = np.where(ecgs['registro'].isin(pacientes_treino), 'treino', 'teste')

    # Salvar em um CSV no caminho correto
    ecgs.to_csv('../database/processed/Dados_Arritmia_Renan.csv', index=False)
    print('Dados tratados exportados com sucesso!')


