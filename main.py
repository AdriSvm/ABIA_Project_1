from abia_energia import *
from math import sqrt
from typing import List, Generator, Set
import timeit
from search import Problem, hill_climbing

class Parameters():
    def __init__(self, n_c: list, n_cl: int, propc: list[float], propg: float, seed:int = 42):
        self.n_c = n_c
        self.n_cl = n_cl
        self.propc = propc
        self.propg = propg
        self.seed = seed

class Operators():
    pass


class SwapClients(Operators):
    def __init__(self, cl1: int, cent1: int, cl2: int, cent2: int):
        self.cl1 = cl1
        self.cl2 = cl2
        self.cent1 = cent1
        self.cent2 = cent2

    def __repr__(self) -> str:
        return f" | Swap clients {self.cl1} from {self.cent1} and {self.cl2} from {self.cent2}"


class SwapState(Operators):
    def __init__(self, ce:int, estate: bool):
        self.ce = ce
        self.estate = estate

    def __repr__(self) -> str:
        return f" | Central {self.ce} changed its state to {self.estate}"


class MoveClient(Operators):
    def __init__(self, cl: int, cent1: int, cent2: int):
        self.cl = cl
        self.cent1 = cent1
        self.cent2 = cent2

    def __repr__(self) -> str:
        return f" | Client {self.cl} changed central {self.cent1} to central{self.cent2}"

def distance(a:tuple,b:tuple) -> float:
    return sqrt(((a[0]-b[0])**2) + ((a[1]-b[1])**2))

def clients_power(client:int, dicc:dict, clients:Clientes, centrals: Centrales, central = None) -> float:
    if central == None:
        for c in dicc:
            if client in dicc[c]:
                central = c
                break
    c_coords = (clients[client].CoordX, clients[client].CoordY)
    cl_coords = (centrals[central].CoordX, centrals[central].CoordY)
    dist = distance(c_coords,cl_coords)
    reduction = VEnergia.loss(dist)
    client_power = clients[client].Consumo +  (clients[client].Consumo * reduction)

    assert client_power >= clients[client].Consumo
    return client_power


def power_left(central:int, dicc:dict, clients:Clientes, centrals:Centrales) -> float:
    power_left = centrals[central].Produccion
    for cl in dicc[central]:
        power_left -= clients_power(cl,dicc,clients,centrals,central)
    #print(power_left)
    return power_left



class StateRepresentation(object):
    def __init__(self, clients:clientes, centrals:centrales, dict:dict ):
        self.clients = clients
        self.centrals = centrals
        self.dict = dict
    def copy(self):
        new_dict = {x:self.dict[x].copy() for x in self.dict}
        return StateRepresentation(self.clients,self.centrals,new_dict)
    def __repr__(self) -> str:
        lst = []
        for i in range(len(self.centrals)):
            lst.append((self.centrals[i], [self.clients[x] for x in self.centrals[i]]))

        return f"Llista de tuples on el primer element és la central i el segon la llista de clients que té assignats: \n {lst}"

    



    def generate_actions(self):
        #Swap central state
        for c in self.dict:
            if len(self.dict[c]) == 0:
                #self.centrals[c].Estado = False
                yield SwapState(c,False)
            else:
                if self.centrals[c].Estado == False:
                    #self.centrals[c].Estado = True
                    yield SwapState(c, True)
                else:
                    exists_granted = False
                    for x in self.dict[c]:
                        if self.clients[x].Contrato == 0 and not exists_granted:
                            exists_granted = True
                            #self.centrals[c].Estado = True
                    if not exists_granted:
                        yield SwapState(c, False)
                    else:
                        yield SwapState(c, True)
                        

                    

        #Move client to another central
        for c in self.dict:
            for cl in self.dict[c]:
                for c1 in self.dict:
                    if c1 != c:
                        pl = power_left(c1, self.dict, self.clients, self.centrals)
                        if clients_power(c1, self.dict, self.clients, self.centrals) <  pl and pl > 0:
                            yield MoveClient(cl,c,c1)
        
        #Echange two clients
        for central in self.dict:
            for client in self.dict[central]:
                for sec_central in self.dict:
                    if sec_central != central:
                        for sec_client in self.dict[sec_central]:
                            if sec_client != client:
                                pl1=power_left(sec_central, self.dict, self.clients, self.centrals)
                                pl2=power_left(central, self.dict, self.clients, self.centrals)
                                if clients_power(client, self.dict, self.clients, self.centrals) < pl1 \
                                    and clients_power(sec_client, self.dict, self.clients, self.centrals) < pl2  and pl1 > 0 and pl2 > 0:
                                    yield SwapClients(client,central,sec_client,sec_central)

    def apply_action(self, action: Operators):
        new_state = self.copy()
        
        if isinstance(action,SwapState):
            ce = action.ce
            estate = action.estate
            new_state.centrals[ce].Estado = estate

        elif isinstance(action,MoveClient):
            cl = action.cl
            ce1 = action.cent1
            ce2 = action.cent2

            new_state.dict[ce1].remove(cl)
            print(ce2)
            new_state.dict[ce2].add(cl)

        elif isinstance(action, SwapClients):
            cl1 = action.cl1
            cl2 = action.cl2
            ce1 = action.cent1
            ce2 = action.cent2

            new_state.dict[ce1].remove(cl1)
            new_state.dict[ce2].remove(cl2)
            new_state.dict[ce1].add(cl2)
            new_state.dict[ce2].add(cl1)
        return new_state

    def heuristic(self) -> float:
        gains = 0
        for c in self.dict:
            for cl in self.dict[c]:
                client = self.clients[cl]
                type = client.Tipo
                consump = client.Consumo
                deal = client.Contrato
                if self.centrals[c].Estado == False:
                    gains -= VEnergia.tarifa_cliente_penalizacion(type) * consump
                    gains -= VEnergia.stop_cost(self.centrals[c].Tipo)
                else:
                    if deal == 0:
                        gains += VEnergia.tarifa_cliente_garantizada(type) * consump
                    else:
                        gains += VEnergia.tarifa_cliente_no_garantizada(type) * consump

                    gains -= VEnergia.daily_cost(self.centrals[c].Tipo)
                    gains -= VEnergia.costs_production_mw(self.centrals[c].Tipo) * self.centrals[c].Produccion

        return gains



def generate_initial_state(params: Parameters) -> StateRepresentation:
    clients = Clientes(params.n_cl,params.propc,params.propg,params.seed)
    centrals = Centrales(params.n_c,params.seed)
    state_dict = {i: set() for i in range(len(centrals))}

    for c in centrals:
        c.Estado = True

    i = 0
    c = 0
    while c <= len(centrals)-1 and i <= len(clients)-1:
        full = False
        while not full and i <= len(clients)-1:
            if power_left(c,state_dict,clients,centrals) < clients_power(i,state_dict,clients,centrals,c):
                full = True
            else:
                state_dict[c].add(i)
                i += 1
        c += 1

    return StateRepresentation(clients,centrals,state_dict)


class CentralDistributionProblem(Problem):
    def __init__(self, initial_state: StateRepresentation):
        super().__init__(initial_state)

    def actions(self, state: StateRepresentation) -> Generator[Operators, None, None]:
        return state.generate_actions()

    def result(self, state: StateRepresentation, action: Operators) -> StateRepresentation:
        return state.apply_action(action)

    def value(self, state: StateRepresentation) -> float:
        return state.heuristic()

    def goal_test(self, state: StateRepresentation) -> bool:
        return False


initial_state = generate_initial_state(Parameters([1, 4, 5],20, [0.2, 0.3, 0.5], 0.5, 42))
n = hill_climbing(CentralDistributionProblem(initial_state))
print(n)





