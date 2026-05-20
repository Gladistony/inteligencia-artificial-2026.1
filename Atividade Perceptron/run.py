# =========================================================
# Aluno: Gladistony Silva Lins
# GitHub: https://github.com/Gladistony/inteligencia-artificial-2026.1
# Data: 2025-05-20
# =========================================================
# CONSTANTES
# =========================================================
X_TREINO = [
    [1, 0, 0], 
    [1, 0, 1], 
    [1, 1, 0], 
    [1, 1, 1]  
]
Y_TREINO = [0, 0, 1, 1]  
PESOS_INICIAIS = [0.0, 0.0, 0.0]
TAXA_APRENDIZADO = 0.1
CICLOS = 2
CASAS_DECIMAIS = 2
# =========================================================
def funcao_de_ativacao(x):
    return 1 if x >= 0 else 0

def metodo_perceptron(X, Y, w, alpha, ciclos):
    if len(X) != len(Y):
        raise ValueError("O número de exemplos (X) deve ser igual ao número de rótulos (Y).")
    if len(w) != len(X[0]):
        raise ValueError("O número de pesos (w) deve ser igual ao número de atributos em cada exemplo (X).")
    
    w_atual = w.copy()
    historico = []
    
    for ciclo in range(1, ciclos+1):
        for i in range(len(X)):
            x = X[i]
            y_esperado = Y[i]
            
            soma = sum(w_atual[j] * x[j] for j in range(len(w_atual)))
            y_pred = funcao_de_ativacao(soma)
            erro = y_esperado - y_pred
            
            if erro != 0:
                for j in range(len(w_atual)):
                    w_atual[j] = round(w_atual[j] + alpha * erro * x[j], CASAS_DECIMAIS)
            
            historico.append({
                'ciclo': ciclo,
                'exemplo': x,
                'esperado': y_esperado,
                'soma': round(soma, CASAS_DECIMAIS),
                'pred': y_pred,
                'erro': erro,
                'pesos': w_atual.copy()
            })
            
    return w_atual, historico
# =========================================================
if __name__ == "__main__":
    pesos_finais, historico = metodo_perceptron(X_TREINO, Y_TREINO, PESOS_INICIAIS, TAXA_APRENDIZADO, CICLOS)
    
    print("\n" + "="*85)
    print(f"{'TREINAMENTO DO PERCEPTRON':^85}")
    print("="*85)
    
    # Cabeçalho da Tabela
    print(f"| {'Ciclo':^5} | {'Entrada (x)':^15} | {'Esperado':^8} | {'Soma':^6} | {'Predito':^7} | {'Erro':^6} | {'Pesos (w)':^17} |")
    print("-" * 85)
    
    # Linhas da Tabela
    ciclo_atual = 1
    for etapa in historico:
        # Adiciona uma divisória a cada mudança de ciclo para ficar mais legível
        if etapa['ciclo'] != ciclo_atual:
            print("-" * 85)
            ciclo_atual = etapa['ciclo']
            
        print(f"| {etapa['ciclo']:^5} | {str(etapa['exemplo']):^15} | {etapa['esperado']:^8} | {etapa['soma']:^6} | {etapa['pred']:^7} | {etapa['erro']:^6} | {str(etapa['pesos']):^17} |")
    
    # selecionar todos os erros do ultimo ciclo
    erros_ultimo_ciclo = [etapa['erro'] for etapa in historico if etapa['ciclo'] == CICLOS]
    sucesso = all(erro == 0 for erro in erros_ultimo_ciclo)
    
    print("="*85)
    print(f"\nPesos finais após {CICLOS} ciclos: {pesos_finais}")
    if sucesso:
        print("O perceptron convergiu com sucesso!")
    else:
        print("O perceptron não convergiu completamente, mas os pesos foram atualizados.")
