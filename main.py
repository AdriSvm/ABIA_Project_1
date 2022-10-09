from xmlrpc.client import Boolean
from abia_energia import *
from math import sqrt
from typing import List, Generator, Set
import timeit
from search import Problem, hill_climbing, simulated_annealing, exp_schedule

class Parameters():
    def __init__(self, n_c: list, n_cl: int, propc: List[float], propg: float, seed:int = 42):
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

class InsertClient(Operators):
    def __init__(self,cl:int,c:int):
        self.cl = cl
        self.c = c

    def __repr__(self) -> str:
        return f" | Client {self.cl} insertat a la central {self.c}"

class SwapState(Operators):
    def __init__(self, ce:int, estate: bool):
        self.ce = ce
        self.estate = estate

    def __repr__(self) -> str:
        return f" | Central {self.ce} changed its state to {self.estate} AAAAAAAAAAAA"


class MoveClient(Operators):
    def __init__(self, cl: int, cent1: int, cent2: int):
        self.cl = cl
        self.cent1 = cent1
        self.cent2 = cent2

    def __repr__(self) -> str:
        return f" | Client {self.cl} changed central {self.cent2} to central {self.cent1}"

class EchangeClients(Operators):
        def __init__(self,c:int,cl:int,other:list):
            self.c = c
            self.cl = cl
            self.other = other

        def __repr__(self) -> str:
            return f" | Client {self.cl} echanged for {self.other} in central {self.other}"

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
    def __init__(self, clients:clientes, centrals:centrales, dict:dict, states:list, left:list = []):
        self.clients = clients
        self.centrals = centrals
        self.dict = dict
        self.left = left
        self.states = states

    def copy(self):
        new_dict = {x:self.dict[x].copy() for x in self.dict}
        new_left = [x for x in self.left]
        new_states = [x for x in self.states]
        return StateRepresentation(self.clients,self.centrals,new_dict,new_states, new_left)

    def __repr__(self) -> str:
        lst = []
        for i in range(len(self.centrals)):
            #lst.append((self.centrals[i], [self.clients[x] for x in self.dict[i]]))
            lst.append((i, [x for x in self.dict[i]]))
            self.gains = self.heuristic()
        return f"Llista de tuples on el primer element és la central i el segon la llista de clients que té assignats: \n {lst} " \
               f"\n i té uns beneficis de {self.gains}" \
               f"\n Els clients que no estàn assignats a cap central són: \n {len(self.left)}"



    def generate_one_action(self):
        #Swap central state
        swap_state_comb = set()
        for c in self.dict:
            exist_granted = False
            for cl in self.dict[c]:
                if self.clients[cl].Contrato == 0:
                    exist_granted = True
                    break

            if exist_granted:
                swap_state_comb.add((c, True))

            else:
                gains = 0
                for cl in self.dict[c]:
                    gains += VEnergia.tarifa_cliente_no_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo

                gains -= VEnergia.daily_cost(self.centrals[c].Tipo)
                gains -= VEnergia.costs_production_mw(self.centrals[c].Tipo) * self.centrals[c].Produccion
                
                if gains < 0:
                    if self.states[c] == False:
                        pass
                    else:
                        swap_state_comb.add((c, False))
                else:
                    swap_state_comb.add((c, True))

        #Introduce client into set:
        intro_cl_comb = set()
        for c in self.dict:
            for cl in self.left:
                if power_left(c,self.dict,self.clients,self.centrals) > clients_power(cl,self.dict,self.clients,self.centrals,c):
                    intro_cl_comb.add((cl,c))





        #Move client to another central
        move_cl_comb = set()
        for c in self.dict:
            gains_c = 0
            for cl in self.dict[c]:
                if self.clients[cl].Contrato == 0:
                    gains_c += VEnergia.tarifa_cliente_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo

                else:
                    gains_c += VEnergia.tarifa_cliente_no_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo

            for cl in self.dict[c]:
                gains_cm = gains_c
                if self.clients[cl].Contrato == 0:
                    gains_cm -= VEnergia.tarifa_cliente_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo

                else:
                    gains_cm -= VEnergia.tarifa_cliente_no_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo

                gain_cliente = gains_c - gains_cm

                for c1 in self.dict:
                    gains_c1 = 0
                    if c1 != c:
                        for cl in self.dict[c1]:
                            if self.clients[cl].Contrato == 0:
                                gains_c1 += VEnergia.tarifa_cliente_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo

                            else:
                                gains_c1 += VEnergia.tarifa_cliente_no_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo

                        gains_c1 += gain_cliente
                        pl = power_left(c1, self.dict, self.clients, self.centrals)
                        if clients_power(c1, self.dict, self.clients, self.centrals) <  pl and pl > 0 and gains_c1 > gains_c:
                            move_cl_comb.add((cl,c1,c))

        m = len(move_cl_comb)
        i = len(intro_cl_comb)
        s = len(swap_state_comb)

        random_value = random.random()
        if random_value < (m / (i + m + s)):
            combination = random.choice(list(move_cl_comb))
            yield MoveClient(combination[0], combination[1], combination[2])

        elif (m / (i + m + s)) <= random_value <= (i / (i + m + s)):
            combination = random.choice(list(intro_cl_comb))
            yield InsertClient(combination[0], combination[1])  

        else:
            combination = random.choice(list(swap_state_comb))
            yield SwapState(combination[0], combination[1])                  
        
        '''
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
        '''


    def generate_actions(self):

        #Introduce client into set:
        i = 0
        for cl in self.left:
            intro = False
            while i < len(centrales) and not intro:
                if power_left(i,self.dict,self.clients,self.centrals) > clients_power(cl,self.dict,self.clients,self.centrals,i):
                    intro = True
                
                else:
                    i += 1
            if intro:
                yield InsertClient(cl,i)
                    

        #Swap central state
        """
        for c in self.dict:
            if self.states[c] == False:
                #self.states[c] = True
                yield SwapState(c, True)
            else:
                exists_granted = False
                for x in self.dict[c]:
                    if self.clients[x].Contrato == 0 and not exists_granted:
                        exists_granted = True
                        #self.states[c] = True
                if not exists_granted:
                    c_gain = 0
                    for i in self.dict[c]:
                        c_gain +=VEnergia.tarifa_cliente_no_garantizada(self.clients[i].Tipo) * self.clients[i].Consumo
                    c_gain -= VEnergia.daily_cost(self.centrals[c].Tipo)
                    c_gain -= VEnergia.costs_production_mw(self.centrals[c].Tipo) * self.centrals[c].Produccion
                    if c_gain < 0:
                        yield SwapState(c,False)"""


        
        #Move client to another central
        for c in self.dict:
            gains_c = 0
            for cl in self.dict[c]:
                if self.clients[cl].Contrato == 0:
                    gains_c += VEnergia.tarifa_cliente_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo

                else:
                    gains_c += VEnergia.tarifa_cliente_no_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo

            for cl in self.dict[c]:
                gains_cm = gains_c
                if self.clients[cl].Contrato == 0:
                    gains_cm -= VEnergia.tarifa_cliente_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo

                else:
                    gains_cm -= VEnergia.tarifa_cliente_no_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo

                gain_cliente = gains_c - gains_cm

                for c1 in self.dict:
                    gains_c1 = 0
                    if c1 != c:
                        for cl1 in self.dict[c1]:
                            if self.clients[cl].Contrato == 0:
                                gains_c1 += VEnergia.tarifa_cliente_garantizada(self.clients[cl1].Tipo) * self.clients[cl1].Consumo

                            else:
                                gains_c1 += VEnergia.tarifa_cliente_no_garantizada(self.clients[cl1].Tipo) * self.clients[cl1].Consumo

                        gains_c1 += gain_cliente
                    pl = power_left(c1, self.dict, self.clients, self.centrals)
                    if clients_power(cl, self.dict, self.clients, self.centrals,c1) <  pl and pl > 0 and gains_c1 > gains_c:
                        yield MoveClient(cl,c,c1)



        #modificación swap central state
        for c in self.dict:
            exist_granted = False
            for cl in self.dict[c]:
                if self.clients[cl].Contrato == 0 and not exist_granted:
                    exist_granted = True
                    
            gains = 0
            for cl in self.dict[c]:
                gains += VEnergia.tarifa_cliente_no_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo

            gains -= VEnergia.daily_cost(self.centrals[c].Tipo)
            gains -= VEnergia.costs_production_mw(self.centrals[c].Tipo) * self.centrals[c].Produccion
            
            if gains < 0:
                if self.states[c] == False:
                    if exist_granted:
                        yield SwapState(c, True)
                    pass
                else:
                    coste_enc = VEnergia.daily_cost(self.centrals[c].Tipo) + (VEnergia.costs_production_mw(self.centrals[c].Tipo) * self.centrals[c].Produccion)
                    coste_ap = VEnergia.stop_cost(self.centrals[c].Tipo)
                    if coste_ap < coste_enc and not exist_granted:
                        yield SwapState(c, False)
                    else:
                        yield SwapState(c, True)
            else:
                yield SwapState(c, True)
        """
        #Echange client w/ central with another w/o central
        for cl in self.left:
            for c in self.dict:
                not_granted = []
                for cl2 in self.dict[c]:
                    if self.clients[cl2].Contrato == 1:
                        not_granted.append((clients_power(cl2,self.dict,self.clients,self.centrals,c),cl2))
                not_granted.sort()
                for i in range(len(not_granted)):
                    not_granted[i] = not_granted[i][1]
                cl_power = clients_power(cl,self.dict,self.clients,self.centrals,c)
                cl2_power = 0
                i = 0
                while cl2_power + power_left(c,self.dict,self.clients,self.centrals) < cl_power and i < len(not_granted):
                    cl2_power += clients_power(not_granted[i],self.dict,self.clients,self.centrals,c)
                    #print(cl2_power + power_left(c, self.dict, self.clients, self.centrals) >= cl_power)
                    if cl2_power + power_left(c, self.dict, self.clients, self.centrals) >= cl_power:
                        gains_cl2 = 0
                        for i in not_granted[:i]:
                            gains_cl2 += VEnergia.tarifa_cliente_no_garantizada(self.clients[i].Tipo)

                        if VEnergia.tarifa_cliente_no_garantizada(self.clients[cl].Tipo) > gains_cl2:

                            yield EchangeClients(c,cl,not_granted[:i])
                    i += 1
        """
        '''            
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
        '''


    def apply_action(self, action: Operators):
        new_state = self.copy()
        
        if isinstance(action,SwapState):
            ce = action.ce
            estate = action.estate
            new_state.states[ce] = estate

        elif isinstance(action,MoveClient):
            cl = action.cl
            ce1 = action.cent1
            ce2 = action.cent2
            

            new_state.dict[ce1].remove(cl)
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

        elif isinstance(action, InsertClient):
            cl = action.cl
            c = action.c

            
            new_state.left.remove(cl)
            new_state.dict[c].add(cl)
            

        elif isinstance(action,EchangeClients):
            cl = action.cl
            c = action.c
            other = action.other

            for i in other:
                new_state.dict[c].remove(i)
                new_state.left.append(i)
                new_state.dict[c].add(cl)

            new_state.sort_left()

        return new_state

    def heuristic(self) -> float:
        self.gains = 0

        for c in self.dict:
            for cl in self.dict[c]:
                client = self.clients[cl]
                type = client.Tipo
                consump = client.Consumo
                deal = client.Contrato

                if self.states[c] == False:
                    self.gains -= VEnergia.tarifa_cliente_penalizacion(type) * consump
                else:
                    if deal == 0:
                        self.gains += VEnergia.tarifa_cliente_garantizada(type) * consump
                    else:
                        self.gains += VEnergia.tarifa_cliente_no_garantizada(type) * consump

            if self.states[c] == False:
                self.gains -= VEnergia.stop_cost(self.centrals[c].Tipo)
            else:
                self.gains -= VEnergia.daily_cost(self.centrals[c].Tipo)
                self.gains -= VEnergia.costs_production_mw(self.centrals[c].Tipo) * self.centrals[c].Produccion

        for cl in self.left:
            self.gains -= VEnergia.tarifa_cliente_penalizacion(self.clients[cl].Tipo) * self.clients[cl].Consumo

        #print(self.gains)
        return self.gains

def generate_initial_state(params: Parameters) -> StateRepresentation:
    clients = Clientes(params.n_cl,params.propc,params.propg,params.seed)
    centrals = Centrales(params.n_c,params.seed)
    state_dict = {i: set() for i in range(len(centrals))}
    states = [True for x in range(len(centrals))]

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
    print("generao")
    return StateRepresentation(clients,centrals,state_dict,states)

def gen_mitad(params : Parameters) -> StateRepresentation:
    clients = Clientes(params.n_cl, params.propc, params.propg, params.seed)
    centrals = Centrales(params.n_c, params.seed)
    state_dict = {i: set() for i in range(len(centrals))}
    states = [True for x in range(len(centrals))]
    left = []

    i = 0
    for c in state_dict:
        power_central_half = power_left(c, state_dict, clients, centrals)
        power_acum_client = 0
        while power_central_half > power_acum_client and i < len(clients):
            if clients[i].Contrato == 0:
                state_dict[c].add(i)
                power_acum_client += clients_power(i, state_dict, clients, centrals, c)
            else:
                left.append(i)
            i += 1

    return StateRepresentation(clients, centrals, state_dict,states,left)



def generate_initial_state_granted(params: Parameters) -> StateRepresentation:
    clients = Clientes(params.n_cl, params.propc, params.propg, params.seed)
    clients_granted = []
    clients_no_granted = []
    centrals = Centrales(params.n_c, params.seed)
    state_dict = {i: set() for i in range(len(centrals))}
    state = [True for i in range(len(centrals))]
    #state = [False for i in range(len(centrals))]

    for cl in range(len(clients)):
        if clients[cl].Contrato == 0:
            clients_granted.append((cl,clients[cl]))
        else:
            clients_no_granted.append((cl,clients[cl]))


    end = False
    while len(clients_granted) > 0 and not end:
        c = 0
        placed = False
        while c < len(centrals) and not placed:
            if power_left(c, state_dict, clients, centrals) < clients_power(clients_granted[0][0], state_dict, clients, centrals, c):
                c += 1
                if c == len(centrals)-1:
                    end = True
            else:
                state_dict[c].add(clients.index(clients_granted[0][1]))
                c += 1
                placed = True
                clients_granted.pop(0)
    
    end = False
    while len(clients_no_granted) > 0 and not end:
        c = 0
        placed = False
        while c < len(centrals) and not placed:
            if power_left(c, state_dict, clients, centrals) < clients_power(clients_no_granted[0][0], state_dict, clients, centrals, c):
                c += 1
                if c == len(centrals)-1:
                    end = True
            else:
                state_dict[c].add(clients.index(clients_no_granted[0][1]))
                c += 1
                placed = True
                clients_no_granted.pop(0)
    
    if clients_granted != []:
        raise Exception("Not a valid initial state")
    if clients_no_granted != []:
        aux = []
        for cl in clients_no_granted:
            aux.append([cl[1].Consumo,cl[0],cl[1]])
        aux.sort()

        for i in aux[::-1]:
            clients_no_granted.remove((i[1],i[2])) #clients_no_granted.index((i[1],i[2]))
            clients_no_granted.insert(0,i[0])

        return StateRepresentation(clients, centrals, state_dict,state,clients_no_granted)

    return StateRepresentation(clients, centrals, state_dict,state)


class CentralDistributionProblem(Problem):
    def __init__(self, initial_state: StateRepresentation, use_one_action: Boolean = False):
        self.action = use_one_action
        super().__init__(initial_state)

    def actions(self, state: StateRepresentation) -> Generator[Operators, None, None]:
        if self.action:
            return state.generate_one_action()
        
        else:
            return state.generate_actions()

    def result(self, state: StateRepresentation, action: Operators) -> StateRepresentation:
        return state.apply_action(action)

    def value(self, state: StateRepresentation) -> float:
        return state.heuristic()

    def goal_test(self, state: StateRepresentation) -> bool:
        return False





def experiment1():
    
    initial_state = gen_mitad(Parameters([5, 10, 25],1000, [0.2, 0.3, 0.5], 0.5, 22))
    #initial_state = gen_mitad(Parameters([1, 4, 5],100, [0.2, 0.3, 0.5], 0.5, 22))
    initial_gains = initial_state.heuristic()

    n = hill_climbing(CentralDistributionProblem(initial_state))
    print(f" Els gains inicials són : {initial_gains}")

    para_sim_an = [7, 0.0005, 250]
    #n = simulated_annealing(CentralDistributionProblem(initial_state, use_one_action= True), schedule= exp_schedule(k = para_sim_an[0], lam= para_sim_an[1], limit= para_sim_an[2]))
    print(f"La representació del estat inicial és aquesta \n {initial_state}")

    print(f" La representació del estat final és aquesta \n {n}")

    print(f"Els operadors utilitzats han sigut :\n {str(list(n.generate_actions()))}")
    
    #print(f"Els temps és de {timeit.timeit(lambda: hill_climbing(CentralDistributionProblem(generate_initial_state_granted(Parameters([1, 4, 5],100, [0.2, 0.3, 0.5], 0.5, 22)))), number=1)} gen_mitad")
    #print(f"Els temps és de {timeit.timeit(lambda: simulated_annealing(CentralDistributionProblem(generate_initial_state_granted(Parameters([1, 4, 5],100, [0.2, 0.3, 0.5], 0.5, 22))), schedule= exp_schedule(k = para_sim_an[0], lam= para_sim_an[1], limit= para_sim_an[2])), number=1)} granted")


    #print(f"La central 1 tiene un coste encendida de {(VEnergia.costs_production_mw(n.centrals[1].Tipo) * n.centrals[1].Produccion) + VEnergia.daily_cost(n.centrals[1].Tipo)} \n y tiene un coste apagada de {VEnergia.stop_cost(n.centrals[1].Tipo)}")
    #print(timeit.timeit(lambda: simulated_annealing(CentralDistributionProblem(initial_state, use_one_action= True), schedule= exp_schedule(k=1, lam = 0.0005, limit=2500), number=1)))
    #print(f"Els temps és de {timeit.timeit(lambda: hill_climbing(CentralDistributionProblem(gen_mitad(Parameters([5, 10, 25],1000, [0.2, 0.3, 0.5], 0.5, 22)))), number=1)} gen_mitad")
    print(f"Els temps és de {timeit.timeit(lambda: hill_climbing(CentralDistributionProblem(gen_mitad(Parameters([5, 10, 25],1000, [0.2, 0.3, 0.5], 0.5, 22)))), number=1)} granted")
    #print(f"Els temps és de {timeit.timeit(lambda: simulated_annealing(CentralDistributionProblem(generate_initial_state_granted(Parameters([1, 4, 5],100, [0.2, 0.3, 0.5], 0.5, 22))), schedule= exp_schedule(k = para_sim_an[0], lam= para_sim_an[1], limit= para_sim_an[2])), number=1)} granted")


"""
#initial_state = generate_initial_state(Parameters([1, 4, 5],100, [0.2, 0.3, 0.5], 0.5, 22))
initial_state = generate_initial_state_granted(Parameters([5, 10, 25],2000, [0.2, 0.3, 0.5], 0.5, 65))
#initial_state2 = generate_initial_state2(Parameters([5, 10, 25],1000, [0.2, 0.3, 0.5], 0.5, 42))
#print(initial_state)
#print(initial_state2)
initial_gains = initial_state.heuristic()
#initial_gains2 = initial_state2.heuristic()
#n = hill_climbing(CentralDistributionProblem(initial_state))
n = simulated_annealing(CentralDistributionProblem(initial_state, use_one_action= True))
# #n = hill_climbing(CentralDistributionProblem(initial_state2))
print(n)
print(initial_gains)

print(timeit.timeit(lambda: simulated_annealing(CentralDistributionProblem(initial_state, use_one_action= True)), number=1))
#print(timeit.timeit(lambda: hill_climbing(CentralDistributionProblem(initial_state)), number=1))

#print(n)
"""
experiment1()


