
#Operadors
'''
for i, cl in enumerate(self.clients):
    c = 0
    actual = None
    final = None
    if i not in self.left:

        for c in self.dict:
            if i in self.dict[c]:
                actual = c
                break

        miin = VEnergia.loss(
            distance((cl.CoordX, cl.CoordY), (self.centrals[actual].CoordX, self.centrals[actual].CoordY)))

        while c <= len(self.centrals) - 1:
            d = VEnergia.loss(distance((cl.CoordX, cl.CoordY), (self.centrals[c].CoordX, self.centrals[c].CoordY)))
            cons_cl = clients_power(i, self.dict, self.clients, self.centrals, c)

            if miin > d and cons_cl < power_left(c, self.dict, self.clients, self.centrals) and c != actual:
                miin = d
                final = c
            c += 1

        if final != None:
            yield MoveClient(i, actual, final)
            '''
'''#Introduce client into set:
        self.sort_left()
        for c_l in self.left:
            cl_p_init = clients_power(c_l, self.dict, self.clients, self.centrals, 0)
            ce = 0
            for c in self.dict:
                cl_p_actual = clients_power(c_l, self.dict, self.clients,self.centrals, c)
                if power_left(c, self.dict, self.clients, self.centrals) > cl_p_actual and cl_p_actual <= cl_p_init:
                    ce = c

            yield InsertClient(c_l, ce)'''
"""
#Swap central state
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
                yield SwapState(c,False)
"""

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

"""
#Move client
for c1 in self.dict:
    if c1 != c:
        pl = power_left(c1, self.dict, self.clients, self.centrals)
        if clients_power(c1, self.dict, self.clients, self.centrals) <  pl and pl > 0:
            yield MoveClient(cl,c,c1)"""

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
