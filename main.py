from xmlrpc.client import Boolean
from abia_energia import *
from math import sqrt
from typing import List, Generator
import timeit
from search import Problem, hill_climbing, simulated_annealing, exp_schedule


class Parameters:
    """
    Representative class for the problem parameters
    """

    def __init__(self, n_c: list, n_cl: int, propc: List[float], propg: float, seed: int = 42):
        self.n_c = n_c
        self.n_cl = n_cl
        self.propc = propc
        self.propg = propg
        self.seed = seed


class Operators:
    """
    Abstract class
    """
    pass


class InsertClient(Operators):
    """
    Representation of Insert Client operator
    """

    def __init__(self, cl: int, c: int):
        self.cl = cl
        self.c = c

    def __repr__(self) -> str:
        return f" | Client {self.cl} insertat a la central {self.c}"


class SwapState(Operators):
    """
    Representation of Swap State operator
    """

    def __init__(self, ce: int, estate: bool):
        self.ce = ce
        self.estate = estate

    def __repr__(self) -> str:
        return f" | Central {self.ce} changed its state to {self.estate}"


class MoveClient(Operators):
    """
    Representation of Move Client operator
    """

    def __init__(self, cl: int, cent1: int, cent2: int):
        self.cl = cl
        self.cent1 = cent1
        self.cent2 = cent2

    def __repr__(self) -> str:
        return f" | Client {self.cl} changed central {self.cent1} to central {self.cent2}"


class SwapClients(Operators):
    """
    Representation of Exchange Clients operator
    """

    def __init__(self, cl1, c1, cl2, c2):
        self.cl1 = cl1
        self.c1 = c1
        self.cl2 = cl2
        self.c2 = c2

    def __repr__(self) -> str:
        return f" | Client {self.cl1} de la central {self.c1} intercanviat amb {self.cl2} de la central {self.c2}"


def distance(a: tuple, b: tuple) -> float:
    """
    Returns euclidean distance from point a to point b
    """
    return sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))


def clients_power(client: int, dicc: dict, clients: Clientes, centrals: Centrales, central=None) -> float:
    """
    Aux funcion
    Returns the power that central has to produce to satisfy client consumption.
    If not given central finds the client's central assigned, raise exception if not assigned to any central.
    """
    if central is None:
        for c in dicc:
            if client in dicc[c]:
                central = c
                break

    if central is None:
        raise Exception("Central associada al client no trobada")

    c_coords = (clients[client].CoordX, clients[client].CoordY)
    cl_coords = (centrals[central].CoordX, centrals[central].CoordY)
    dist = distance(c_coords, cl_coords)
    reduction = VEnergia.loss(dist)
    client_power = clients[client].Consumo + (clients[client].Consumo * reduction)

    assert client_power >= clients[client].Consumo
    return client_power


def power_left(central: int, dicc: dict, clients: Clientes, centrals: Centrales) -> float:
    """
    Aux function.
    Returns the free power of a central.
    """
    p_l = centrals[central].Produccion
    for cl in dicc[central]:
        p_l -= clients_power(cl, dicc, clients, centrals, central)
    return p_l


class StateRepresentation(object):
    """
    Main class, representation of the problem, its operators and heuristic
    """

    def __init__(self, clients: Clientes, centrals: Centrales, r_dict: dict[int:set], states: list[Boolean],
                 left=None):
        """
        Necessary args for representation
        clients: Clients class with representation of all clients
        centrals: Centrals class with representation of all power plants
        dict: Dictionary, len(dict) = len(centrals), int: set(), 0 to n centrals, each with one set
        States: list[Bool], len(states) = len(centrals), Bool indicates if central n its ON or OFF
        left: list[int], all clients not assigned to a central, there can't be granted clients to assure safe solution
        """
        if left is None:
            left = []
        self.clients = clients
        self.centrals = centrals
        self.dict = r_dict
        self.states = states
        self.left = left
        self.sort_left()
        self.gains = None

    def copy(self):
        """
        Creates a new representation wo/ any link
        """
        new_dict = {x: self.dict[x].copy() for x in self.dict}
        new_left = [x for x in self.left]
        new_states = [x for x in self.states]
        return StateRepresentation(self.clients, self.centrals, new_dict, new_states, new_left)

    def sort_left(self):
        """
        Sorts left clients ordered by the biggest to the smallest consumption
        """
        aux = []
        for cl in self.left:
            aux.append([self.clients[cl].Consumo, cl, self.clients[cl]])
        aux.sort()

        for i in aux[::-1]:
            self.left.remove(i[1])  # clients_no_granted.index((i[1],i[2]))
            self.left.insert(0, i[1])

    def __repr__(self) -> str:
        lst = []
        for i in range(len(self.centrals)):
            lst.append((i, len([x for x in self.dict[i]])))
            self.gains = self.heuristic()
        wo_power = []
        for i in range(len(self.states)):
            if not self.states[i]:
                wo_power.append(i)
        return f"Llista de tuples on el primer element és la central i el segon la quantitat de clients assignats:" \
               f" \n {lst} " \
               f"\n i té uns beneficis de {self.gains}" \
               f"\n Els clients que no estàn assignats a cap central són: \n {len(self.left)} \ni no tenen " \
               f"electricitat:\n {[self.dict[x] for x in wo_power]}"

    def generate_one_action(self):
        """
        Function for Simmulated anealing algorithm
        Implemented operators:
            Insert client:
                Pick up first client of self.left(biggest consumption).
                Find and pick up one of the nearest centrals from the client(cl) which can allocate the client.
                Insert it there.
            Move client:
                For each client asigned to a central, find its central and another where client has lower consumption.
                Reminder: Consumption of a client depends on distance to central, lower clients_power = lower distance.
                If there's a central where consumption is lower, move client from c_init to c_fin.
            Swap state:
                For each central, if there's a granted client, turn it on.
                If there's not, calculate its gains, and turn it on or off if gains are better.
            Exchange Clients:
                For each central and each client, find other n-1 clients and try to exchange each other.
                Not recommendable use for the implemented generative functions *HIGH PERFORMANCE NEEDED*.
            For the results showed in informe.pdf in GitHub, best results with only Insert Client and Swap State
                """
        # Insert Client
        intro_cl_comb = set()
        if len(self.left) > 0:
            cl = self.left[0]
            miin = VEnergia.loss(distance((self.clients[cl].CoordX, self.clients[cl].CoordY),
                                          (self.centrals[0].CoordX, self.centrals[0].CoordY)))
            c = 0
            for i in self.dict:
                d = VEnergia.loss(distance((self.clients[cl].CoordX, self.clients[cl].CoordY),
                                           (self.centrals[i].CoordX, self.centrals[i].CoordY)))

                if miin > d and power_left(i, self.dict, self.clients, self.centrals) >= clients_power(cl, self.dict,
                                                                                                       self.clients,
                                                                                                       self.centrals,
                                                                                                       i):
                    miin = d
                    c = i

            intro_cl_comb.add((cl, c))

        '''# Move client to another central
        move_cl_comb = set()
        for cl in range(len(self.clients)):
            c_fin = None
            if cl not in self.left:
                cons_cl = clients_power(cl, self.dict, self.clients, self.centrals)

                for c in self.dict:
                    if cl in self.dict[c]:
                        c_init = c

                    cons_cl_fin = clients_power(cl, self.dict, self.clients, self.centrals, c)

                    if cons_cl > cons_cl_fin:
                        cons_cl = cons_cl_fin
                        c_fin = c

            if c_fin != None:
                move_cl_comb.add((cl, c_init, c_fin))'''
        # Swap state
        swap_state_comb = set()
        for c in self.dict:
            exist_granted = False
            for cl in self.dict[c]:
                if self.clients[cl].Contrato == 0 and not exist_granted:
                    exist_granted = True

            if not exist_granted:
                gains = 0
                for cl in self.dict[c]:
                    gains += VEnergia.tarifa_cliente_no_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo

                gains -= VEnergia.daily_cost(self.centrals[c].Tipo)
                gains -= VEnergia.costs_production_mw(self.centrals[c].Tipo) * self.centrals[c].Produccion

                if gains < 0:
                    if not self.states[c]:
                        pass
                    else:
                        coste_enc = VEnergia.daily_cost(self.centrals[c].Tipo) + (
                                VEnergia.costs_production_mw(self.centrals[c].Tipo) * self.centrals[c].Produccion)
                        coste_ap = VEnergia.stop_cost(self.centrals[c].Tipo)
                        if coste_ap < coste_enc:
                            swap_state_comb.add((c, False))
                        else:
                            swap_state_comb.add((c, True))
                else:
                    swap_state_comb.add((c, True))
            else:
                if not self.states[c]:
                    swap_state_comb.add((c, True))

        s = len(swap_state_comb)
        i = len(intro_cl_comb)

        random_value = random.random()
        if random_value < (i / (i + s)):
            combination = random.choice(list(intro_cl_comb))
            yield InsertClient(combination[0], combination[1])

        else:
            combination = random.choice(list(swap_state_comb))
            yield SwapState(combination[0], combination[1])

    def generate_actions(self):
        """
        Function for Hill climbing algorithm
        Implemented operators:
            Insert client:
                Pick up first client of self.left(biggest consumption).
                Find and pick up one of the nearest centrals from the client(cl) which can allocate the client.
                Insert it there.
            Move client:
                For each client assigned to a central, find its central and another where client has lower consumption.
                Reminder: Consumption of a client depends on distance to central, lower clients_power = lower distance.
                If there's a central where consumption is lower, move client from c_init to c_fin.
            Swap state:
                For each central, if there's a granted client, turn it on.
                If there's not, calculate its gains, and turn it on or off if gains are better.
            Exchange Clients:
                For each central and each client, find other n-1 clients and try to exchange each other.
                Not recommendable use for the implemented generative functions *HIGH PERFORMANCE NEEDED*.

        For the results showed in informe.pdf in GitHub, best results with only Insert Client and Swap State
        """
        # InsertClient
        if len(self.left) > 0:
            cl = self.left[0]
            minim = VEnergia.loss(distance((self.clients[cl].CoordX, self.clients[cl].CoordY),
                                           (self.centrals[0].CoordX, self.centrals[0].CoordY)))
            c = 0
            for i in self.dict:
                d = VEnergia.loss(distance((self.clients[cl].CoordX, self.clients[cl].CoordY),
                                           (self.centrals[i].CoordX, self.centrals[i].CoordY)))

                if minim > d and power_left(i, self.dict, self.clients, self.centrals) >= \
                        clients_power(cl, self.dict, self.clients, self.centrals, i):
                    minim = d
                    c = i

            yield InsertClient(cl, c)
        '''
        # Move client
        for cl in range(len(self.clients)):
            c_fin = None
            c_init = None
            if cl not in self.left:
                cons_cl = clients_power(cl, self.dict, self.clients, self.centrals)

                for c in self.dict:
                    if cl in self.dict[c]:
                        c_init = c

                    cons_cl_fin = clients_power(cl, self.dict, self.clients, self.centrals, c)

                    if cons_cl > cons_cl_fin:
                        cons_cl = cons_cl_fin
                        c_fin = c

            if c_fin != None and c_init != None:
                yield MoveClient(cl, c_init, c_fin)
        '''

        # Swap State
        for c in self.dict:
            exist_granted = False
            for cl in self.dict[c]:
                if self.clients[cl].Contrato == 0 and not exist_granted:
                    exist_granted = True
                    # If exists granted i central, it MUST be ON
                    self.states[c] = True

            if not exist_granted:
                gains = 0
                ind = 0
                for cl in self.dict[c]:
                    gains += VEnergia.tarifa_cliente_no_garantizada(self.clients[cl].Tipo) * self.clients[cl].Consumo
                    ind += VEnergia.tarifa_cliente_penalizacion(self.clients[cl].Tipo) * self.clients[cl].Consumo

                if self.states[c]:
                    gains -= VEnergia.daily_cost(self.centrals[c].Tipo)
                    gains -= VEnergia.costs_production_mw(self.centrals[c].Tipo) * self.centrals[c].Produccion
                else:
                    gains -= VEnergia.stop_cost(self.centrals[c].Tipo)

                if gains < 0:
                    if not self.states[c]:
                        pass
                    else:
                        coste_enc = VEnergia.daily_cost(self.centrals[c].Tipo) + (
                                VEnergia.costs_production_mw(self.centrals[c].Tipo) * self.centrals[c].Produccion)
                        coste_ap = VEnergia.stop_cost(self.centrals[c].Tipo) + ind

                        if coste_ap < coste_enc:
                            yield SwapState(c, False)
                else:
                    yield SwapState(c, True)

        '''
        # Echange clients
        for central in self.dict:
            for client in self.dict[central]:
                for sec_central in self.dict:
                    if sec_central != central:
                        for sec_client in self.dict[sec_central]:
                            if sec_client != client:
                                pl1 = power_left(sec_central, self.dict, self.clients, self.centrals)
                                pl2 = power_left(central, self.dict, self.clients, self.centrals)
                                if clients_power(client, self.dict, self.clients, self.centrals) < pl1 \
                                        and clients_power(sec_client, self.dict, self.clients,
                                                          self.centrals) < pl2 and pl1 > 0 and pl2 > 0:
                                    yield SwapClients(client, central, sec_client, sec_central)
        '''

    def apply_action(self, action: Operators):
        """
        Function for operator execution.
        Creates new_state, a copy of StateRepresentation and returns it with operator modification.
        Creating successors.
        """

        new_state = self.copy()

        if isinstance(action, SwapState):
            ce = action.ce
            estate = action.estate
            new_state.states[ce] = estate

        elif isinstance(action, MoveClient):
            cl = action.cl
            ce1 = action.cent1
            ce2 = action.cent2

            new_state.dict[ce1].remove(cl)
            new_state.dict[ce2].add(cl)

        elif isinstance(action, InsertClient):
            cl = action.cl
            c = action.c

            new_state.dict[c].add(cl)
            new_state.left.pop(0)

        elif isinstance(action, SwapClients):
            cl1 = action.cl1
            c1 = action.c1
            cl2 = action.cl2
            c2 = action.c2

            new_state.dict[c1].remove(cl1)
            new_state.dict[c2].remove(cl2)
            new_state.dict[c1].add(cl2)
            new_state.dict[c2].add(cl1)

        return new_state

    def heuristic(self) -> float:
        """
        Heuristic function, evaluates successors.
        Returns gains, it can be modified for other options like entropy or assigning weights.
        """
        self.gains = 0

        for c in self.dict:
            for cl in self.dict[c]:
                client = self.clients[cl]
                tp = client.Tipo
                consump = client.Consumo
                deal = client.Contrato

                if not self.states[c]:
                    self.gains -= VEnergia.tarifa_cliente_penalizacion(tp) * consump
                    self.gains += VEnergia.tarifa_cliente_no_garantizada(tp) * consump
                else:
                    if deal == 0:
                        self.gains += VEnergia.tarifa_cliente_garantizada(tp) * consump
                    else:
                        self.gains += VEnergia.tarifa_cliente_no_garantizada(tp) * consump

            if not self.states[c]:
                self.gains -= VEnergia.stop_cost(self.centrals[c].Tipo)
            else:
                self.gains -= VEnergia.daily_cost(self.centrals[c].Tipo)
                self.gains -= VEnergia.costs_production_mw(self.centrals[c].Tipo) * self.centrals[c].Produccion

        for cl in self.left:
            self.gains -= VEnergia.tarifa_cliente_penalizacion(self.clients[cl].Tipo) * self.clients[cl].Consumo
        # print(self.gains)
        return self.gains

    def heuristic_p5(self, x: int = 0) -> float:
        self.gains = 0
        self.x = x

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

        garan = [x for x in filter(lambda x: self.clients[x].Contrato == 0, self.left)]

        return self.gains - self.x * len(garan)

def gen_initial_state_only_granted(params: Parameters) -> StateRepresentation:
    """
    Function for generating the initial state.
    Assigns all guaranteed clients to one of its nearest centrals. Not guaranteed keeps out in left[list]
    Guaranteed clients must fit on all centrals.
    Quite greedy.
    """
    clients = Clientes(params.n_cl, params.propc, params.propg, params.seed)
    centrals = Centrales(params.n_c, params.seed)
    state_dict = {i: set() for i in range(len(centrals))}
    states = [True for _ in range(len(centrals))]
    # granted expression for guaranteed clients, only abbreviation stuff
    clients_granted = []
    clients_no_granted = []

    for cl in range(len(clients)):
        if clients[cl].Contrato == 0:
            clients_granted.append(cl)
        else:
            clients_no_granted.append(cl)

    for cl in clients_granted:
        lossers = []
        c = 0
        while c <= len(centrals) - 1:
            loss = VEnergia.loss(
                distance((clients[cl].CoordX, clients[cl].CoordY), (centrals[c].CoordX, centrals[c].CoordY)))
            lossers.append(loss)
            c += 1

        central = lossers.index(min(lossers))
        state_dict[central].add(cl)

    return StateRepresentation(clients, centrals, state_dict, states, clients_no_granted)


def gen_initial_state_ordered(params: Parameters) -> StateRepresentation:
    """
    Function for generating the initial state.
    Assigns all guaranteed clients in entrance order, then with not guaranteed clients.
    Guaranteed clients must fit on all centrals, if not an exception is raised.
    Not so greedy, but assigning all clients we get a fine initial solution.
    """
    clients = Clientes(params.n_cl, params.propc, params.propg, params.seed)
    clients_granted = []
    clients_no_granted = []
    centrals = Centrales(params.n_c, params.seed)
    state_dict = {i: set() for i in range(len(centrals))}
    state = [True for _ in range(len(centrals))]

    for cl in range(len(clients)):
        if clients[cl].Contrato == 0:
            clients_granted.append((cl, clients[cl]))
        else:
            clients_no_granted.append((cl, clients[cl]))

    end = False
    while len(clients_granted) > 0 and not end:
        c = 0
        placed = False
        while c < len(centrals) and not placed:
            if power_left(c, state_dict, clients, centrals) < \
                    clients_power(clients_granted[0][0], state_dict, clients, centrals, c):
                c += 1
                if c == len(centrals) - 1:
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
            if power_left(c, state_dict, clients, centrals) < \
                    clients_power(clients_no_granted[0][0], state_dict, clients, centrals, c):
                c += 1
                if c == len(centrals) - 1:
                    end = True
            else:
                state_dict[c].add(clients.index(clients_no_granted[0][1]))
                c += 1
                placed = True
                clients_no_granted.pop(0)

    if clients_granted:
        raise Exception("Not a valid initial state")
    if clients_no_granted:
        aux = []
        for cl in clients_no_granted:
            aux.append([cl[1].Consumo, cl[0], cl[1]])
        aux.sort()

        for i in aux[::-1]:
            clients_no_granted.remove((i[1], i[2]))  # clients_no_granted.index((i[1],i[2]))
            clients_no_granted.insert(0, i[1])

        return StateRepresentation(clients, centrals, state_dict, state, clients_no_granted)

    return StateRepresentation(clients, centrals, state_dict, state)


def gen_initial_state_empty(params: Parameters) -> StateRepresentation:
    """
    Function for generating the initial state in experiment 5.
    Only all clients in left, so no client is assigned.
    Zero greedy and initial state it's not a solution for the problem.
    """
    clients = Clientes(params.n_cl, params.propc, params.propg, params.seed)
    centrals = Centrales(params.n_c, params.seed)
    state_dict = {i: set() for i in range(len(centrals))}
    states = [True for _ in range(len(centrals))]
    clients_no_granted = []
    clients_granted = []

    for cl in range(len(clients)):
        if clients[cl].Contrato == 0:
            clients_granted.append(cl)

        else:
            clients_no_granted.append(cl)

    clients_mixed = clients_granted + clients_no_granted

    return StateRepresentation(clients, centrals, state_dict, states, clients_mixed)


class CentralDistributionProblem(Problem):
    """
    Problem subclass.
    use_one_action, True for Simulated annealing algorithm.
    """

    def __init__(self, initial_state: StateRepresentation, use_one_action: Boolean = False, use_p5: Boolean = False, pen: int = 0):
        self.action = use_one_action
        self.use_p5 = use_p5
        self.pen = pen
        super().__init__(initial_state)

    def actions(self, state: StateRepresentation) -> Generator[Operators, None, None]:
        if self.action:
            return state.generate_one_action()

        else:
            return state.generate_actions()

    def result(self, state: StateRepresentation, action: Operators) -> StateRepresentation:
        return state.apply_action(action)

    def value(self, state: StateRepresentation) -> float:
        if self.use_p5:
            return state.heuristic_p5(x=self.pen)
        else:
            return state.heuristic()

    def goal_test(self, state: StateRepresentation) -> bool:
        return False


def experiment(algorithm: str, method: str, n_c: list[int], n_cl: int, propcl: list[float], propg: float, seed: int,
               k=5, limit=750, lam=0.0005, timming=False, n_iter=5, x=0):
    """
    Auxiliary function for making easy the experiments executions.
    USE ONLY FOR HILL CLIMBING AND SIMULATED ANNEALING.
    USED FOR OUR LOCAL EXECUTIONS AND TESTS.
    """
    method = method.upper()
    algorithm = algorithm.upper()
    initial_state = None
    n = None
    if type(timming) == str or not timming:
        if algorithm == 'HILL CLIMBING':

            if method == 'ORDERED':
                initial_state = gen_initial_state_ordered(Parameters(n_c, n_cl, propcl, propg, seed))
                n = hill_climbing(CentralDistributionProblem(initial_state))

            elif method == 'ONLY GRANTED':
                initial_state = gen_initial_state_only_granted(Parameters(n_c, n_cl, propcl, propg, seed))
                n = hill_climbing(CentralDistributionProblem(initial_state))

            elif method == 'PROBLEM 5':
                initial_state = gen_initial_state_empty(Parameters(n_c, n_cl, propcl, propg, seed))
                n = hill_climbing(CentralDistributionProblem(initial_state, False, True, x))
                print(f"La representació del estat inicial és aquesta \n {initial_state}")

                print(f" La representació del estat final és aquesta \n {n}")

        elif algorithm == 'SIMULATED ANNEALING':
            if method == 'ORDERED':
                initial_state = gen_initial_state_ordered(Parameters(n_c, n_cl, propcl, propg, seed))
                n = simulated_annealing(CentralDistributionProblem(initial_state, use_one_action=True),
                                        schedule=exp_schedule(k=k, lam=lam, limit=limit))

            elif method == 'ONLY GRANTED':
                initial_state = gen_initial_state_only_granted(Parameters(n_c, n_cl, propcl, propg, seed))
                n = simulated_annealing(CentralDistributionProblem(initial_state, use_one_action=True),
                                        schedule=exp_schedule(k=k, lam=lam, limit=limit))

            elif method == 'PROBLEM 5':
                print("Calculant algorisme")
                initial_state = gen_initial_state_empty(Parameters(n_c, n_cl, propcl, propg, seed))
                n = simulated_annealing(CentralDistributionProblem(initial_state, True, True, x),
                                        schedule=exp_schedule(k=10, lam=0.005, limit=1000))
                print(f"La representació del estat inicial és aquesta \n {initial_state}")

                print(f" La representació del estat final és aquesta \n {n}")


        if timming:
            timming = True

    if timming:
        if algorithm == 'HILL CLIMBING':
            if method == 'ORDERED':
                return timeit.timeit(lambda: hill_climbing(CentralDistributionProblem(gen_initial_state_ordered
                                                                                      (Parameters(n_c, n_cl, propcl,
                                                                                                  propg, seed)))),
                                     number=n_iter), False

            elif method == 'ONLY GRANTED':
                return timeit.timeit(lambda: hill_climbing(CentralDistributionProblem(gen_initial_state_only_granted(
                    Parameters(n_c, n_cl, propcl, propg, seed)))),
                                     number=n_iter), False

            elif method == 'PROBLEM 5':
                # print(
                #    f'''Els temps és de {timeit.timeit(lambda: hill_climbing(CentralDistributionProblem
                #    (gen_initial_state_only_granted(Parameters([5, 10, 25], 1000, [0.2, 0.3, 0.5], 0.5, 22)))), number=n_iter)}''')
                return timeit.timeit(lambda: hill_climbing(CentralDistributionProblem
                                                           (gen_initial_state_empty(Parameters(n_c, n_cl, propcl, propg, seed)), False, True,
                                                            x)), number=n_iter), False

        elif algorithm == 'SIMULATED ANNEALING':
            if method == 'ORDERED':
                return timeit.timeit(lambda: simulated_annealing(CentralDistributionProblem(gen_initial_state_ordered(
                    Parameters(n_c, n_cl, propcl, propg, seed)), use_one_action=True),
                    schedule=exp_schedule(k=k, lam=lam, limit=limit)),
                                     number=n_iter), False

            elif method == 'ONLY GRANTED':
                return timeit.timeit(
                    lambda: simulated_annealing(CentralDistributionProblem(gen_initial_state_only_granted(
                        Parameters(n_c, n_cl, propcl, propg, seed)), use_one_action=True),
                        schedule=exp_schedule(k=k, lam=lam, limit=limit)),
                    number=n_iter), False

            elif method == 'PROBLEM 5':
                print("calculant temps")
                # print(
                #    f'''Els temps és de {timeit.timeit(lambda: simulated_annealing(CentralDistributionProblem
                #    (gen_initial_state_only_granted(Parameters([5, 10, 25], 1000, [0.2, 0.3, 0.5], 0.5, 22)))), number=n_iter)}''')
                return timeit.timeit(lambda: simulated_annealing(CentralDistributionProblem
                                                                 (gen_initial_state_empty(Parameters(n_c, n_cl, propcl, propg, seed)), True,
                                                                  True, x),
                                                                 schedule=exp_schedule(k=10, lam=0.005, limit=1000)),
                                     number=n_iter), False

    if initial_state is None or n is None:
        raise Exception("Not executed")
    return initial_state, n


def main():
    initial_state, n = experiment('HILL CLIMBING', 'PROBLEM 5', [5, 10, 25], 1000, [0.2, 0.3, 0.5], 0.75, 22, x=2000)
    print(initial_state, n)

if __name__ == "__main__":
    main()
