import importlib.machinery
from tabulate import tabulate
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def get_col_mapped(K, OX):
    return K * OX;

def get_row_mapped(C, FX, FY, OX):

    overlap = (FX - 1)/FX
    return C * FX * FY * OX * (1 - overlap)

def get_comp_cycles(r_dict, c_dict, layer):

    return layer['OY']*(layer['K']/c_dict['K'])*\
        (layer['OX']/c_dict['OX'])*(layer['FX']/r_dict['FX'])*\
        (layer['FY']/r_dict['FY'])*(layer['C']/r_dict['C'])

def get_utilization(rows, cols, r_dict, c_dict):

    ut_over = r_dict['FX']/(r_dict['FX']+c_dict['OX']-1)
    mapped_utilization = get_row_mapped(r_dict['C'],r_dict['FX'],r_dict['FY'],c_dict['OX'])*\
        get_col_mapped(c_dict['K'],c_dict['OX']) * ut_over / (rows*cols)
    return mapped_utilization

def get_weight_writes(c_dict, r_dict, layer):
    return (layer['FX']/r_dict['FX'])*(layer['FY']/r_dict['FY'])*\
        (layer['C']/r_dict['C'])*(layer['K']/c_dict['K'])
    
if __name__ == "__main__":
    layer_path = "cost_model_input/layer/"
    layer_filename = "DarkNet19"
    rows = 1024
    cols = 512
    layer_spec = importlib.machinery.SourceFileLoader('%s' % (layer_filename),
                                                      '%s%s.py' % (layer_path, layer_filename)).load_module()

    ii_l = 1;
    layers = [1,2,4,5,8]
    layers = [1,2,3,5,6,8,9,11,13,14,16,18]
    layers = [7]

    for ii_l in layers:
        l = layer_spec.layer_info[ii_l]
        col_lpf = []
        row_lpf = []
        for x in l:
            if x in ['K','OX']:
                lpf_list = prime_factors(l[x])
                for lpf in lpf_list:
                    col_lpf.append(tuple([x,lpf]))
            if x in ['C','FX','FY']:
                lpf_list = prime_factors(l[x])
                for lpf in lpf_list:
                    row_lpf.append(tuple([x,lpf]))
        print(row_lpf, col_lpf)

        for k_col in range(0, len(col_lpf)):
            col_combs = set(combinations(col_lpf, k_col))
            for cc in col_combs:
                if np.prod([x[1] for x in cc]) <= cols:
                    for r_col in range(0, len(row_lpf)):
                        row_combs = set(combinations(row_lpf, r_col))
                        for rc in row_combs:
                            r_dict = {'C' : 1, 'FX': 1, 'FY' : 1}
                            for r in r_dict:
                                if r in [x[0] for x in rc]:
                                    r_dict[r] = np.prod([x[1] for x in rc if x[0] == r])
                            c_dict = {'K' : 1, 'OX' : 1}
                            for c in c_dict:
                                if c in [x[0] for x in cc]:
                                    c_dict[c] = np.prod([x[1] for x in cc if x[0] == c])
                            row_mapped = get_row_mapped(r_dict['C'],r_dict['FX'],r_dict['FY'],c_dict['OX'])
                            col_mapped = get_col_mapped(c_dict['K'],c_dict['OX'])
                            print('COL COMB: ', c_dict)
                            print('ROW COMB: ', r_dict)
                            print('ROW MAPPED: ', row_mapped)
                            print('COL MAPPED: ', col_mapped)
                            print('CYCLES: ', get_comp_cycles(r_dict, c_dict, l))
                            print('UT: ', get_utilization(rows, cols, r_dict, c_dict))
                            print()





































        
    # fig_col, ax_col = plt.subplots()
    # fig_row, ax_row = plt.subplots()
    # fig_cyc, ax_cyc = plt.subplots()
    # fig_w, ax_w = plt.subplots()
    # fig_u, ax_u = plt.subplots()
    # w = 0.3
    
    # for iii_l, ii_l in enumerate(darknet19_layers):
    #     l = layer_spec.layer_info[ii_l]
    #     table_spec_cm = {}
    #     table_spec_cm["col_mapped"] = l['K']
    #     table_spec_cm["row_mapped"] = l['C']*l['FY']*(l['FX'])
    #     table_spec_cm["comp_cycles"] = l['OY']* l['OX']
    #     table_spec_cm["weight_rewrite"] = 1
    #     table_spec_opu = {}
    #     table_spec_opu["col_mapped"] = l['K']*l['OX']
    #     table_spec_opu["row_mapped"] = l['C']*l['FY']*(l['FX']*2 - 1)
    #     table_spec_opu["comp_cycles"] = l['OY']
    #     table_spec_opu["weight_rewrite"] = 1
    #     table_spec_cm["ut"] = 1
    #     table_spec_opu["ut"] = 3/5
        
    #     # print("CM: LAYER ",ii_l,":",table_spec_cm)
    #     # print("OPU: LAYER ",ii_l,":",table_spec_opu)
    #     # cm_table = [[x, table_spec_cm[x]] for x in table_spec_cm]
    #     # print(tabulate(cm_table))

    #     max_col = 512

    #     max_row = 1024
    
    #     # print("ROW LIMIT: 1024")
    #     l = layer_spec.layer_info[ii_l]
    #     table_spec_opu2 = {}
    #     fit_row = False
    #     fit_col = False
    #     fit = False
    #     ox_div = 1
    #     c_div = 1
    #     k_div = 1
    #     while not fit:
    #         if l['FX'] != 1:
    #             table_spec_opu2["col_mapped"] = l['K']*l['OX'] / (ox_div*k_div)
    #             if ox_div == l['OX']:
    #                 table_spec_opu2["row_mapped"] = l['C']*l['FY']*l['FX']/c_div
    #                 table_spec_opu2["ut"] = table_spec_opu2["col_mapped"] * table_spec_opu2["row_mapped"] / (1024*512)
                
    #             elif ox_div >= 2*l['OX']:
    #                 table_spec_opu2["row_mapped"] = l['C']*l['FY']*(l['FX']*2 - 2)/c_div
    #                 table_spec_opu2["ut"] = table_spec_opu2["col_mapped"] * table_spec_opu2["row_mapped"] * 0.75 / (1024*512)
    #             else:
    #                 table_spec_opu2["row_mapped"] = l['C']*l['FY']*(l['FX']*2 - 1)/c_div
    #                 table_spec_opu2["ut"] = table_spec_opu2["col_mapped"] * table_spec_opu2["row_mapped"] * 0.6 / (1024*512)
                                                                
    #             table_spec_opu2["comp_cycles"] = l['OY'] * ox_div * c_div * k_div
                
    #             table_spec_opu2["weight_rewrite"] = c_div * k_div
    #         else:
    #             table_spec_opu2["col_mapped"] = l['K']*l['OX'] / (ox_div*k_div)
    #             if ox_div == l['OX']:
    #                 table_spec_opu2["row_mapped"] = (l['C']*l['FY']*l['FX']/c_div)
    #                 table_spec_opu2["ut"] = table_spec_opu2["col_mapped"] * table_spec_opu2["row_mapped"] / (1024*512)
                
    #             else:
    #                 table_spec_opu2["row_mapped"] = (l['C']*l['FY']*(l['FX'])/c_div)*(l['OX']/ox_div)
    #                 table_spec_opu2["ut"] = table_spec_opu2["col_mapped"] * table_spec_opu2["row_mapped"] * 0.6 / (1024*512)
                                                                
    #             table_spec_opu2["comp_cycles"] = l['OY'] * ox_div * c_div * k_div
                
    #             table_spec_opu2["weight_rewrite"] = c_div * k_div

                
    #         if table_spec_opu2["col_mapped"] <= max_col and table_spec_opu2["col_mapped"]%1 == 0:
    #             fit_col = True
    #         else:
    #             if ox_div < l['OX']:
    #                 ox_div += 1
    #             else:
    #                 k_div += 1
    #                 while l['K']/k_div % 1 != 0:
    #                     k_div += 1
    #         if table_spec_opu2["row_mapped"] <= max_row and table_spec_opu2["row_mapped"]%1 == 0:
    #             fit_row = True
    #         else:
    #             c_div += 1
    #             while l['C']/c_div % 1 != 0:
    #                 c_div += 1
    #         if fit_row and fit_col:
    #             fit = True
    #     print(ii_l, table_spec_opu2, c_div, k_div, ox_div)
                
    #     l = layer_spec.layer_info[ii_l]
    #     table_spec_cm2 = {}
    #     fit_row = False
    #     fit_col = False
    #     fit = False
    #     k_div = 1
    #     c_div = 1
    #     while not fit:
    #         table_spec_cm2["col_mapped"] = l['K']/k_div
    #         table_spec_cm2["row_mapped"] = l['C']*l['FY']*l['FX']/c_div
    #         table_spec_cm2["comp_cycles"] = l['OY']*l['OX']* k_div * c_div
    #         table_spec_cm2["weight_rewrite"] = c_div*k_div
    #         table_spec_cm2["ut"] = table_spec_cm2["col_mapped"] * table_spec_cm2["row_mapped"] / (1024*512)
    #         if table_spec_cm2["col_mapped"] <= max_col and table_spec_cm2["col_mapped"]%1 == 0:
    #             fit_col = True
    #         else:
    #             k_div += 1
    #             while l['K']/k_div % 1 != 0:
    #                 k_div += 1
                
    #         if table_spec_cm2["row_mapped"] <= max_row and table_spec_cm2["row_mapped"]%1 == 0:
    #             fit_row = True
    #         else:
    #             c_div += 1
    #             while l['C']/c_div % 1 != 0:
    #                 c_div += 1
    #         if fit_row and fit_col:
    #             fit = True
        
        
        
    #     col_map = ["col_mapped",table_spec_cm["col_mapped"],table_spec_opu["col_mapped"],table_spec_cm2["col_mapped"],table_spec_opu2["col_mapped"]]
    #     row_map = ["row_mapped",table_spec_cm["row_mapped"],table_spec_opu["row_mapped"],table_spec_cm2["row_mapped"],table_spec_opu2["row_mapped"]]
    #     cycles = ["comp_cycles",table_spec_cm["comp_cycles"],table_spec_opu["comp_cycles"],table_spec_cm2["comp_cycles"],table_spec_opu2["comp_cycles"]]
    #     weight_w = ["weight_rewrite",table_spec_cm["weight_rewrite"],table_spec_opu["weight_rewrite"],table_spec_cm2["weight_rewrite"],table_spec_opu2["weight_rewrite"]]
    #     ut = ["utilization",table_spec_cm["ut"],table_spec_opu["ut"],table_spec_cm2["ut"],table_spec_opu2["ut"]]
    #     print("LAYER ",ii_l)
    #     print("LAYER_SPEC: K",l['K'], "C:",l['C'], "OX:",l['OX'], "OY:",l['OY'], "FX:",l['FX'], "FY:",l['FY'])
    #     print(tabulate([col_map, row_map,cycles,weight_w,ut], headers=["CM","OPU","CM 1024x512","OPU 1024x512"], tablefmt="presto"))
    #     print()
    #     # if iii_l == 0:
    # #         ax_col.bar(iii_l, table_spec_cm2['col_mapped'],w,color="teal",label="Conventional")
    # #         ax_col.bar(iii_l+w, table_spec_opu2['col_mapped'],w,color="orange",label="O.P.U.")
    # #         ax_row.bar(iii_l, table_spec_cm2['row_mapped'],w,color="teal",label="Conventional")
    # #         ax_row.bar(iii_l+w, table_spec_opu2['row_mapped'],w,color="orange",label="O.P.U.")
    # #         ax_w.bar(iii_l, table_spec_cm2['weight_rewrite'],w,color="teal",label="Conventional")
    # #         ax_w.bar(iii_l+w, table_spec_opu2['weight_rewrite'],w,color="orange",label="O.P.U.")
    # #         ax_u.bar(iii_l, table_spec_cm2['ut'],w,color="teal",label="Conventional")
    # #         ax_u.bar(iii_l+w, table_spec_opu2['ut'],w,color="orange",label="O.P.U.")
    # #         ax_cyc.bar(iii_l, table_spec_cm2['comp_cycles'],w,color="teal",label="Conventional")
    # #         ax_cyc.bar(iii_l+w, table_spec_opu2['comp_cycles'],w,color="orange",label="O.P.U.")
    # #     else:
    # #         ax_row.bar(iii_l, table_spec_cm2['row_mapped'],w,color="teal")
    # #         ax_row.bar(iii_l+w, table_spec_opu2['row_mapped'],w,color="orange")
    # #         ax_w.bar(iii_l, table_spec_cm2['weight_rewrite'],w,color="teal")
    # #         ax_w.bar(iii_l+w, table_spec_opu2['weight_rewrite'],w,color="orange")
    # #         ax_u.bar(iii_l, table_spec_cm2['ut'],w,color="teal")
    # #         ax_u.bar(iii_l+w, table_spec_opu2['ut'],w,color="orange")
    # #         ax_cyc.bar(iii_l, table_spec_cm2['comp_cycles'],w,color="teal")
    # #         ax_cyc.bar(iii_l+w, table_spec_opu2['comp_cycles'],w,color="orange")
            
    # # ax_col.legend()
    # # ax_col.set_ylabel("Columns mapped")
    # # ax_col.set_xticks(range(0,len(darknet19_layers)))
    # # ax_col.set_xticklabels([str(i) for i in darknet19_layers])
    # # fig_col.tight_layout()
    # # fig_col.savefig("col.png")

    # # ax_row.legend()
    # # ax_row.set_ylabel("Rows mapped")
    # # ax_row.set_xticks(range(0,len(darknet19_layers)))
    # # ax_row.set_xticklabels([str(i) for i in darknet19_layers])
    # # fig_row.tight_layout()
    # # fig_row.savefig("row.png")

    # # ax_u.legend()
    # # ax_u.set_ylabel("Utilization")
    # # ax_u.set_xticks(range(0,len(darknet19_layers)))
    # # ax_u.set_xticklabels([str(i) for i in darknet19_layers])
    # # fig_u.tight_layout()
    # # fig_u.savefig("u.png")

    # # ax_w.legend()
    # # ax_w.set_ylabel("Weight rewrites")
    # # ax_w.set_xticks(range(0,len(darknet19_layers)))
    # # ax_w.set_xticklabels([str(i) for i in darknet19_layers])
    # # fig_w.tight_layout()
    # # fig_w.savefig("w.png")

    # # ax_cyc.legend()
    # # ax_cyc.set_ylabel("Computing cycles")
    # # ax_cyc.set_xticks(range(0,len(darknet19_layers)))
    # # ax_cyc.set_xticklabels([str(i) for i in darknet19_layers])
    # # fig_cyc.tight_layout()
    # # fig_cyc.savefig("cyc.png")

        

        
