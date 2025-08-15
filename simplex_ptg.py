import numpy as np
import sympy as sp

def simplex(Z, A, b, C_inicial, Bi=None, tipo="min", fase_uno=0):
    C = C_inicial.copy()
    # Esto para el 2 fases pues de fase 1 pasa la matriz base
    if Bi is None:
        Bi = np.eye(len(C_inicial)) # Matriz identidad rango len(C)
    iteracion = 0
    max_iter = 100
    # Creamos array simbólico de las variables
    aux = len(Z)
    VB = [sp.Symbol(f'x_{i}') for i in range(1, aux+1)]
    #print(f"Variables = {VB}")

    x_alternativa = None

    while True:
        # Imprimir iteración
        #print(f"\n--- Iteración {iteracion} ---\n")

        # Variables
        #print("Variables básicas:", [VB[i] for i in C])
        #print("Variables no básicas:", [VB[i] for i in range(A.shape[1]) if i not in C])

        # Prueba de optimalidad
        # Se obtinene los coeficientes básicos, no básicos y las columnas no básicas
        cb = Z[C]
        #print("Coeficientes básicos (cb):", cb)
        #print("Matriz inversa:\n", np.round(Bi, 2))
        no_basicas = [j for j in range(A.shape[1]) if j not in C]
        Pj = A[:, no_basicas]
        Cj = Z[no_basicas]
        #print("Columnas no básicas (Pj):\n", Pj)
        #print("Coeficientes no básicos (Cj):", Cj)
        # La prueba: (coef.básicos @ Bi @ columnas.no_básicas) - coeficientes.no_básicos
        cb = cb.reshape(1, -1)
        opt = (cb @ Bi) @ Pj - Cj
        #print("Prueba de optimalidad (opt):", np.round(opt, 2))

        # Verificar optimalidad
        # El if te junta las 2 funciones en una
        # En maximización cuando ya no hay valores negativos
        # En minimización cuando ya no hay valores positivos
        if (tipo == "max" and np.all(opt >= 0)) or (tipo == "min" and np.all(opt <= 0)):
            print("\n !!!!!! EL ALGORITMO TERMINÓ !!!!!! \n")
            x_basicos = Bi @ b
            x = np.zeros(A.shape[1])
            for i, idx in enumerate(C):
                x[idx] = x_basicos[i]
            #print("\nSolución óptima:")
            #for i, var in enumerate(VB):
                #print(f"{var}: {x[i]:.2f}")
            #print(f"Valor óptimo Z: {np.dot(Z, x):.2f}")

            # fase_uno == 0 indica que no estamos en fase 1 por lo que ES NECESARIO
            # ver si existen soluciones alternativas
            if fase_uno == 0:
              if np.any(opt == 0):
                print('\n== Hay indicadores con 0, sugiere soluciones alternativas ==\n')
                opt_aux = opt.flatten()
                for j in range(len(opt_aux)):
                  v_aux = no_basicas[j]  # Índice de variable entrante auxiliar
                  P_aux = A[:, v_aux].reshape(-1, 1) # Columnas no básicas auxiliares
                  alpha_aux = Bi @ P_aux
                  numeradores_aux = Bi @ b.reshape(-1, 1)
                  # Theta auxiliar
                  with np.errstate(divide='ignore'):
                    theta_aux = np.where(alpha_aux > 0, numeradores_aux / alpha_aux, -1)
                    theta_aux[theta_aux < 0] = 777_000_000

                  if np.all(theta_aux >= 777_000_000):
                    continue

                  w_aux = np.argmin(theta_aux) # Índice variale saliente auxiliar
                  # Calcular nueva matriz inversa
                  epsilon_aux = np.zeros_like(alpha_aux)
                  epsilon_aux[w_aux] = 1 / alpha_aux[w_aux]
                  filtro_aux = np.arange(len(alpha_aux)) != w_aux
                  epsilon_aux[filtro_aux] = -alpha_aux[filtro_aux] / alpha_aux[w_aux]
                  E_aux = np.eye(len(C))
                  E_aux[:, w_aux] = epsilon_aux.flatten()
                  Bi_aux = E_aux @ Bi
                  # Actualizar variables básicas auxiliares
                  C_aux = C.copy()
                  C_aux[w_aux] = v_aux
                  # x_alternativa
                  x_basicos_aux = Bi_aux @ b
                  x_alternativa = np.zeros(A.shape[1])
                  for i_aux, idx_aux in enumerate(C_aux):
                      x_alternativa[idx_aux] = x_basicos_aux[i_aux]
                  return {'C': C,'Bi': Bi,'x': x,'opt_val': np.dot(Z, x),'x_alternativa': x_alternativa}

              print("No se encontraron soluciones alternativas factibles")

            break

        # Si no alcanzó el óptimo, determinar variable entrante
        if tipo == "max":
            indice_entrante = np.argmin(opt)# En maximización entra el más negativo
        else:
            indice_entrante = np.argmax(opt)# En minimización entra el más positivo
        v = no_basicas[indice_entrante]
        #print(f"\nVariable entrante: {VB[v]} (índice {v})")

        # Vector alpha
        P = A[:, v].reshape(-1, 1)
        alpha = Bi @ P
        #print("Vector alpha:\n", alpha)

        # Numeradores
        numeradores = Bi @ b.reshape(-1, 1)
        #print("Numeradores:\n", numeradores)

        # Theta
        with np.errstate(divide='ignore'):
            theta = np.where(alpha > 0, numeradores / alpha, -1)# Sólo divide positivos, pone -1 en lo demás
        theta[theta < 0] = 777_000_000 # Pone 777 millones en donde no es válido

        # CASO ESPECIAL: región no acotada
        if np.all(theta >= 777_000_000):
          print("\n==== REGIÓN FACTIBLE NO ACOTADA ====")
          return None

        #print("Theta:\n", np.round(theta, 2))

        # Variable saliente
        w = np.argmin(theta)
        #print(f"\nVariable saliente: {VB[C[w]]} (posición {w} en C)")

        # Actualización bariables básicas
        C[w] = v
        #print("Nuevas variables básicas:", [VB[i] for i in C])

        # Actualización invera de la base
        # Vector epsilon y matriz E
        epsilon = np.zeros_like(alpha) # Copiamos alpha
        epsilon[w] = 1 / alpha[w] # Pivote
        filtro = np.arange(len(alpha)) != w # Filtro, 1º copiamos forma y 2ºhacemos False en pivote
        epsilon[filtro] = -alpha[filtro] / alpha[w] # Operación sólo en valores True
        #print("Vector epsilon:\n", np.round(epsilon, 2))
        E = np.eye(len(C))
        E[:, w] = epsilon.flatten()
        #print("Matriz E:\n", np.round(E, 2))
        # Actualizar matriz inversa
        Bi = E @ Bi
        #print("Nueva matriz inversa:\n", np.round(Bi, 2))

        # Tope de iteraciones
        iteracion += 1
        if iteracion >= max_iter:
            print("Máximo de iteraciones alcanzado")
            break

    # Calcular solución final y devolver resultados
    x_basicos = Bi @ b
    x = np.zeros(A.shape[1])
    for i, idx in enumerate(C):
        x[idx] = x_basicos[i]
    return {'C': C, 'Bi': Bi, 'x': x, 'opt_val': np.dot(Z, x), 'x_alternativa': x_alternativa}

def simplex_dos_fases(Z_original, A_original, b, vars_artificiales, tipo_original="max"):
    # Fase 1
    # Minimizamos suma de variables artificiales
    # (i.e. todos los coeficientes de Z son 0 menos los de las artificiales)
    # Los indicadores de columna de las vars artificiales son los básicos de fase 1 !!!!
    #print("\n=== FASE 1 ===")
    Z_f1 = np.zeros_like(Z_original) # Copia forma de Z, todos 0
    Z_f1[vars_artificiales] = 1  # Coeficientes 1 para artificiales
    # Devolvemos indicadores de variables básicas (C), Bi, vector x_i con valores de variables, y valor óptimo
    resultado_f1 = simplex(Z_f1, A_original, b, C_inicial=vars_artificiales, tipo="min", fase_uno=1)

    # Checa si existe región factible o si está acotada
    if resultado_f1 is None:
      print("\n==== REGIÓN FACTIBLE NO ACOTADA====")
      return None
    elif resultado_f1['opt_val'] != 0:
      print("\n==== NO EXISTE REGIÓN FACTIBLE ====")
      return None

    # Fase 2
    #print("\n=== FASE 2 ===")
    # Eliminar columnas de variables artificiales
    columnas_a_mantener = [i for i in range(A_original.shape[1]) if i not in vars_artificiales]
    A_f2 = A_original[:, columnas_a_mantener]
    Z_f2 = Z_original[columnas_a_mantener]

    # Ajustar índices
    # Tenemos que eliminar lo de las variables artificiales
    C_f2 = [idx for idx in resultado_f1['C'] if idx not in vars_artificiales]

    # Pasar la matriz inversa de la Fase 1 a la Fase 2
    Bi_f2 = resultado_f1['Bi']

    # Hacer fase 2
    resultado_f2 = simplex(Z_f2, A_f2, b, C_inicial=C_f2, Bi=Bi_f2, tipo=tipo_original, fase_uno=0)
    return resultado_f2

def lector(A, Z, condiciones):
    # Mapeamos las condiciones a un array
    mapeo = {'l': 1, 'g': -1, 'i': 0}
    array_holgura = np.array([mapeo[cond] for cond in condiciones])

    # Guardamos indices de variables estructurales
    indices_estructurales = list(range(A.shape[1]))

    # Caso simplex normal
    if np.all(array_holgura == 1):
        # Hacemos matriz holguras y pegamos a A
        holguras = np.eye(A.shape[0]) * array_holgura[:A.shape[0]]
        A = np.hstack((A, holguras))
        #print('matriz A',A)

        # Obtenemos incdices auxiliares (esto para función simplex)
        indices_aux = list(range(A.shape[1] - len(array_holgura), A.shape[1]))
        #print('indices auxiliares',indices_aux)

        # Ajustamos Z (esto para función simplex)
        Z = np.concatenate((Z, np.zeros(len(array_holgura))))
        #print('Z',Z)

    # Caso simplex 2 fases
    else:
        # El mapeo es el mismo
        mapeo = {'l': 1, 'g': -1, 'i': 0}
        array_holgura = np.array([mapeo[cond] for cond in condiciones])

        # En artificiales, si tenemos 1 en holgura, pasa a 0; 1 en los otros casos
        array_artificiales = np.where(array_holgura == 1, 0, 1)

        # Guardamos indices de variables estructurales
        indices_estructurales = list(range(A.shape[1]))

        # Hacemos matrices holguras y artificailes, y pegamos a A
        holguras = np.eye(len(array_holgura)) * array_holgura
        artificiales = np.eye(len(array_artificiales)) * array_artificiales
        A = np.hstack((A, holguras, artificiales))

        # Ajustamos Z (esto para función simplex)
        Z = np.concatenate((Z, np.zeros(len(array_holgura) + len(array_artificiales))))

        # Obtenemos incdices auxiliares (esto para función simplex)
        indices_aux = list(range(A.shape[1] - len(array_artificiales), A.shape[1]))

    return A, Z, indices_estructurales, indices_aux, array_holgura

def Simplex(A, Z, b, condiciones, tipo='max'):
  A = np.array(A)
  Z = np.array(Z)
  b = np.array(b)
  x_alternativa = None
  A, Z, indices_estructurales, indices_aux, array_holgura = lector(A, Z, condiciones)
  # Pasamos array holgura para saber si todas las condiciones son ≤ (\leq) := 1
  # en caso de que sí, usamos simplex normal
  if np.all(array_holgura == 1):
    res = simplex(Z, A, b, C_inicial=indices_aux, tipo=tipo)
  else:
    res = simplex_dos_fases(Z, A, b, indices_aux, tipo_original=tipo)

  if res is None: # Para casos donde el espacio no existe o no es acotado
    return
  elif res['x_alternativa'] is not None: # Para casos donde hay soluciones múltiples
    vector_soluciones = res['x']
    valores_estruturales = vector_soluciones[indices_estructurales]
    valor_optimo = res['opt_val']
    print(f"Valores óptimos: {np.round(valores_estruturales, 2)}")
    print(f"Z: {valor_optimo}")
    #print("----Z = ",np.round(np.dot(Z[:len(valores_estruturales)], valores_estruturales), 2))
    vector_soluciones_alt = res['x_alternativa']
    valores_estruturales_alt = vector_soluciones_alt[indices_estructurales]
    if np.dot(Z[:len(valores_estruturales)], valores_estruturales) != np.dot(Z[:len(valores_estruturales)], valores_estruturales_alt):
      print("\nNo se encontraron soluciones alternativas factibles")
      return
    print(f"Valores óptimos alternativos: {np.round(valores_estruturales_alt, 2)}")
    #print("----Z = ",np.round(np.dot(Z[:len(valores_estruturales)], valores_estruturales_alt), 2))
    #print("\nEspacio solución: λ(valores óptimos) + (1-λ)(valores óptimos alternativos)\n")
    return
  else: # Para casos donde sólo hay una solución
    vector_soluciones = res['x'] # Este tiene TODAS las soluciones x_i, s_i, A_i
    valores_estruturales = vector_soluciones[indices_estructurales] # Sacamos sólo las x_i
    valor_optimo = res['opt_val']
    print(f"Valores óptimos: {np.round(valores_estruturales, 2)}")
    print(f"Z: {valor_optimo}")
    return