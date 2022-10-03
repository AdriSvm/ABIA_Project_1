import random
from math import trunc
from typing import List

"""
Ejemplos de uso:
centrales = Centrales([5, 10, 25], 42)
clientes = Clientes(1000, [0.2, 0.3, 0.5], 0.5, 42)
ganancia_0 = clientes[i].Consumo * VEnergia.tarifa_cliente_garantizada(CLIENTE_MG)

Podéis encontrar constantes que se corresponden al enunciado,
como por ejemplo tipos de central o cliente, al final de este fichero.
"""


class Centrales(list):
    """
    Vector de centrales de energía
    Parámetros de la constructora:
    * centrales_por_tipo: Vector con tres posiciones con el número de centrales
                          de cada tipo a generar
    * seed:               Semilla para el generador de números aleatorios
    La constructora lanza una excepción si el vector de centrales no tiene tamaño 3
    """

    def __init__(self, centrales_por_tipo: List[int], seed: int):
        if len(centrales_por_tipo) != 3:
            raise Exception("Vector Centrales de tamaño incorrecto")
        v_centrales = []
        rand = random.Random(seed)
        for i in range(3):
            for j in range(centrales_por_tipo[i]):
                p = (rand.random() * PROD[i][0]) + PROD[i][1]
                c = Central(TIPO[i], trunc(p), rand.randint(0, 100), rand.randint(0, 100))
                v_centrales.append(c)
        super().__init__(v_centrales)


class Clientes(list):
    """
    Vector de clientes
    Parámetros de la constructora:
    * ncl    : Número de clientes
    * propc  : Proporción de los tipos de clientes (vector 3 posiciones, suman 1)
    * propg  : Proporción de clientes con servicio garantizado
    * seed   : Semilla del generador de números aleatorios
    La constructora lanza una excepción si los vectores de proporciones no
    cumplen las restricciones.
    """

    def __init__(self, ncl: int, propc: List[float], propg: float, seed: int):
        if len(propc) != 3:
            raise Exception("Vector proporciones tipos clientes de tamaño incorrecto")
        if (propc[0] + propc[1] + propc[2]) != 1.0:
            raise Exception("Vector proporciones tipos clientes no suma 1")
        if 0.0 > propg > 1.0:
            raise Exception("Proporcion garantizado fuera de limites")

        v_clientes = []
        rand = random.Random(seed + 1)

        for i in range(ncl):
            dice = rand.random()
            if dice < propc[0]:
                rd = 0
            elif dice < (propc[0] + propc[1]):
                rd = 1
            else:
                rd = 2

            dice = rand.random()
            if dice < propg:
                rc = 0
            else:
                rc = 1

            c = (rand.random() * CONSUMOS[rd][0]) + CONSUMOS[rd][1]
            v_clientes.append(
                Cliente(TIPOCL[rd], trunc(c), TIPOCNT[rc], rand.randint(0, 100), rand.randint(0, 100)))
        super().__init__(v_clientes)


class VEnergia(object):
    """
    Clase estática que permite acceder a los valores de los
    costes de las centrales y las tarifas y penalizaciones
    de los clientes.
    """

    @staticmethod
    def tarifa_cliente_garantizada(tipo: int) -> float:
        """
        Retorna la tarifa de un cliente con servicio garantizado.
        El parámetro tipo especifica el tipo del cliente.
        Lanza una excepción si el tipo está fuera de rango.
        """
        if tipo < 0 or tipo > 2:
            raise Exception("Tipo fuera de rango")
        else:
            return PRECIOS[tipo][0]

    @staticmethod
    def tarifa_cliente_no_garantizada(tipo: int) -> float:
        """
        Retorna la tarifa de un cliente con servicio no garantizado.
        El parámetro tipo especifica el tipo del cliente.
        Lanza una excepción si el tipo está fuera de rango.
        """
        if tipo < 0 or tipo > 2:
            raise Exception("Tipo fuera de rango")
        else:
            return PRECIOS[tipo][1]

    @staticmethod
    def tarifa_cliente_penalizacion(tipo: int) -> float:
        """
        Retorna la penalización por servir a un cliente con
        servicio no garantizado.
        El parámetro tipo especifica el tipo del cliente.
        Lanza una excepción si el tipo está fuera de rango.
        """
        if tipo < 0 or tipo > 2:
            raise Exception("Tipo fuera de rango")
        else:
            return PRECIOS[tipo][2]

    @staticmethod
    def costs_production_mw(tipo: int) -> float:
        """
        Retorna el coste de producción por MW para una central
        de un tipo.
        El parámetro tipo especifica el tipo de la central.
        Lanza una excepción si el tipo está fuera de rango.
        """
        if tipo < 0 or tipo > 2:
            raise Exception("Tipo fuera de rango")
        else:
            return COSTES[tipo][0]

    @staticmethod
    def daily_cost(tipo: int) -> float:
        """
        Retorna el coste de una central en marcha segun su tipo.
        El parámetro tipo especifica el tipo de la central.
        Lanza una excepción si el tipo está fuera de rango.
        """
        if tipo < 0 or tipo > 2:
            raise Exception("Tipo fuera de rango")
        else:
            return COSTES[tipo][1]

    @staticmethod
    def stop_cost(tipo: int) -> float:
        """
        Retorna el coste de una central en parada segun su tipo.
        El parámetro tipo especifica el tipo de la central.
        Lanza una excepción si el tipo está fuera de rango.
        """
        if tipo < 0 or tipo > 2:
            raise Exception("Tipo fuera de rango")
        else:
            return COSTES[tipo][2]

    @staticmethod
    def loss(distancia: float) -> float:
        """
        Retorna la perdida (en tanto por uno) segun la distancia
        entre central y cliente.
        El parámetro distancia especifica la distancia entre central y cliente.
        """
        i = 0
        while distancia > PERDIDA[i][0]:
            i = i + 1
        return PERDIDA[i][1]


class Central(object):
    """
    Características de la central de energía
    """

    def __init__(self, tipo: int, produccion: float, cx: int, cy: int):
        self.Tipo = tipo  # Tipo de central
        self.Produccion = produccion  # Producción en MW
        self.CoordX = cx  # Coordenada x
        self.CoordY = cy  # Coordenada y

    def __repr__(self) -> str:
        return f"Central(tipo={self.Tipo}|produccion={self.Produccion}|cx={self.CoordX}|cy={self.CoordY})"


class Cliente(object):
    """
    Características del cliente
    """

    def __init__(self, t: int, cons: float, cont: int, cx: int, cy: int):
        self.Tipo = t  # Tipo del cliente
        self.Consumo = cons  # Consumo demandado
        self.Contrato = cont  # Tipo de contrato
        self.CoordX = cx  # Coordenada x
        self.CoordY = cy  # Coordenada y

    def __repr__(self) -> str:
        return f"Cliente(Tipo={self.Tipo}|Consumo={self.Consumo}|" +\
               f"Contrato={self.Contrato}|CoordX={self.CoordX}|CoordY={self.CoordY})"


"""
Constantes comunes a todos los escenarios
"""
# Tipos de cliente
CLIENTE_XG = 0
CLIENTE_MG = 1
CLIENTE_G = 2

# Tipos de central
CENTRAL_A = 0
CENTRAL_B = 1
CENTRAL_C = 2

# Prioridades
GARANTIZADO = 0
NOGARANTIZADO = 1

# Producción por tipo de central
PROD = [[500.0, 250.0], [150.0, 100.0], [90.0, 10.0]]

# Consumo por tipo de cliente
CONSUMOS = [[15.0, 5.0], [3.0, 2.0], [2, 1]]

# Tabla de precios por tipo de cliente
# Primera dimensión: tipo de cliente
# Segunda dimensión, por orden: garantizado, no garantizado, indemnización
PRECIOS = [[40.0, 30.0, 5.0], [50.0, 40.0, 5.0], [60.0, 50.0, 5.0]]

# Tabla de costes
# Primera dimensión: tipo de central
# Segunda dimensión, por orden: coste por producción, coste diario, coste parada
COSTES = [[5, 2000, 1500], [8, 1000, 500], [15, 500, 150]]

# Tabla de pérdidas
# En cada posición, el primer valor es la distancia máxima del rango, el segundo valor es el tanto por uno de pérdida.
PERDIDA = [[10, 0], [25, 0.1], [50, 0.2], [75, 0.4], [1000, 0.6]]
TIPO = [0, 1, 2]
TIPOCL = [0, 1, 2]
TIPOCNT = [0, 1]
