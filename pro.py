import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from flask import Flask, request,jsonify

CPUs = pd.read_csv("cpufinalv.csv")
GPU = pd.read_csv("gpus.csv")
CPUs['Price']= CPUs["Price"].replace(',','',regex=True).astype(float)
GPU['Price']= GPU["Price"].replace(',','',regex=True).astype(float)
CPUclean = CPUs.copy()
remove_spec = lambda a : a.strip()
create_list = lambda a : list(map(remove_spec,re.split('& |,|\\*|-', a)))
for col in ['Categories','SubCategories'] :
    CPUclean[col] = CPUclean[col].apply(create_list)
for col in ['Category','Subcategory'] :
    GPU[col] = GPU[col].apply(create_list) 

def cleaner(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
for col in ['Categories','SubCategories'] :
    CPUclean[col] = CPUclean[col].apply(cleaner)
for col in ['Category','Subcategory'] :
    GPU[col] = GPU[col].apply(cleaner)
    
def couple(x):
    return ' '.join(x['Categories']) + ' ' + ' '.join(x['SubCategories'])
def coupleGPU(x):
    return ' '.join(x['Category']) + ' ' + ' '.join(x['Subcategory'])
CPUclean['Features'] = CPUclean.apply(couple, axis=1)
GPU['Features'] = GPU.apply(coupleGPU, axis=1)


count_vector = CountVectorizer(stop_words="english")
matrix = count_vector.fit_transform(CPUclean['Features'])
matrixGPU = count_vector.fit_transform(GPU['Features'])
Cosine_sim =  cosine_similarity(matrix,matrix)
Cosine_sim_GPU = cosine_similarity(matrixGPU,matrixGPU)
Cosine_sim_df = pd.DataFrame(Cosine_sim)
Cosine_sim_df_GPU = pd.DataFrame(Cosine_sim_GPU)

def CPU_Recomm_System_v4(title, max_price):
    a=CPUclean.copy()
    matches = process.extractOne(title, a['Features'])
    if matches[1]<1:
       return 'No Close match found'
    index = a[a['Features'] == matches[0]].index[0]
    top_n_index = list(Cosine_sim_df[index].nlargest(20).index)
    try:
        top_n_index.remove(index)
    except:
        pass
    if  max_price is not None:
        filtered_indices = []
        for idx in top_n_index:
            price = float(a.loc[idx, 'Price'])  
            if (price <= max_price):
                filtered_indices.append(idx)
        top_n_index = filtered_indices
    similar_df = a.iloc[top_n_index][['CpuName', 'Price', 'SingleScore', 'MultiScore']]
    similar_df['cosine_similarity'] = Cosine_sim_df[index].iloc[top_n_index]
    similar_df = similar_df.sort_values(by=['SingleScore', 'MultiScore'], ascending=False)
    return similar_df
 
def GPU_Recomm_System(title,max_price):
    a=GPU.copy()
    
    matches = process.extractOne(title, a['Features'])
    if matches[1]<1:
       return 'No Close match found'
    index = a[a['Features'] == matches[0]].index[0]
    top_n_index = list(Cosine_sim_df_GPU[index].nlargest(20).index)
    try:
        top_n_index.remove(index)
    except:
        pass
    if  max_price is not None:
        filtered_indices = []
        for idx in top_n_index:
            price = float(a.loc[idx, 'Price'])  
            if (price <= max_price):
                filtered_indices.append(idx)
        top_n_index = filtered_indices
    similar_df = a.iloc[top_n_index][['GpuName', 'Price', 'G3DMark', 'G2DMark']]
    similar_df['cosine_similarity'] = Cosine_sim_df_GPU[index].iloc[top_n_index]
    similar_df = similar_df.sort_values(by=['G3DMark', 'G2DMark'], ascending=False)
    return similar_df

def Get_precentage(inputStr):
    if inputStr == "workstation":
        return 30
    elif inputStr == "desktop":
        return 20
    else:
        return 25
        
MB = pd.read_csv("MB.csv")        
def Get_MB(budget,formfactor):
    filtered_MB = MB[(MB["FormFactor"] == formfactor) & (MB["Price"] <= int(budget))]
    sorted_MB = filtered_MB.sort_values(by='Price', ascending = False)
    if sorted_MB.empty:
        return "no matches for SSD"
    return sorted_MB.iloc[0][["Name","Price","FormFactor"]].tolist()

RAM = pd.read_csv("RAM.csv")
def Get_RAM(budget):
    filtered_RAM = RAM[RAM["Price"]<=int(budget)]
    sorted_RAM = filtered_RAM.sort_values(by="Price",ascending=False)
    if sorted_RAM.empty:
        return "no matches for SSD"
    return sorted_RAM.iloc[0][["Name","Price","Size"]].tolist()

HDD=pd.read_csv("HDD.csv")
def Get_HDD(budget):
    filtered_HDD = HDD[HDD["Price"]<=int(budget)]
    sorted_HDD = filtered_HDD.sort_values(by="Price",ascending=False)
    if sorted_HDD.empty:
        return "no matches for HDD"
    return sorted_HDD.iloc[0][["Name","Price","Size"]].tolist()

SSD = pd.read_csv("SSD.csv")
def Get_SSD(budget):
    filtered_SSD = SSD[SSD["Price"]<=int(budget)]
    sorted_SSD = filtered_SSD.sort_values(by="Price",ascending=False)
    if sorted_SSD.empty:
        return "no matches for SSD"
    return sorted_SSD.iloc[0][["Name","Price","Size"]].tolist()

cases = pd.read_csv("modified_case.csv")
def Get_cases(budget,type):
    condition = ""
    if type=="tower":
        condition="ATX Full Tower"
    elif type == "mid tower":
        condition="MicroATX Mid Tower"
    else:
        condition="Mini ITX Tower"
    filtered_case = cases[(cases["type"] == condition)&(cases["price"]<=int(budget))]
    sorted_cases = filtered_case.sort_values(by="price",ascending=False)
    if sorted_cases.empty:
        return "no matches for cases"
    return sorted_cases.iloc[0][["name","price","type"]].tolist()

psu=pd.read_csv("modified_psu.csv")
def Get_psu(budget):
    filtered_psu = psu[psu["price"]<=int(budget)]
    sorted_psu = filtered_psu.sort_values(by="price",ascending=False)
    if sorted_psu.empty:
        return "no matches for SSD"
    return sorted_psu.iloc[0][["name","price"]].tolist()

cooler=pd.read_csv("modified_cooler.csv")
def Get_cooler(budget):
    filtered_cooler = cooler[cooler["price"]<=int(budget)]
    sorted_cooler = filtered_cooler.sort_values(by="price",ascending=False)
    if sorted_cooler.empty:
        return "no matches for cooler"
    return sorted_cooler.iloc[0][["name","price"]].tolist()


        
    
app = Flask(__name__)

@app.route("/recommsys/<input>,<p1>,<SSDcheck>,<case>")
def RecommSys(input,p1,SSDcheck,case):
    percentage = Get_precentage(input.split()[0])
    budget = int(p1)
    CPUprice=budget*(percentage)/100
    GPUprice =budget*25/100
    MBprice = budget*10/100
    RAMprice = budget*10/100
    SSDprice=0
    HDDprice=0
    if(SSDcheck == "yes"):
        SSDprice = budget*10/100
    else:
        SSDprice = budget*5/100
        HDDprice = budget*5/100
    Caseprice = budget*10/100
    PSUprice = budget*8/100
    Coolerprice = budget*4/100
    MBform = ""
    if(case == "tower"):
        MBform = "ATX"
    elif(case == "mid tower"):
        MBform = "Micro-ATX"
    elif(case == "mini tower"):
        MBform = "Mini-ITX"
    
    
    
    MBresult = Get_MB(MBprice,MBform)
    RAMresult = Get_RAM(RAMprice)
    HDDresult = Get_HDD(HDDprice)
    SSDresult= Get_SSD(SSDprice)
    Caseresult = Get_cases(Caseprice,case)
    PSUresult = Get_psu(PSUprice)
    Coolerresult = Get_cooler(Coolerprice)
    
    recommend = CPU_Recomm_System_v4(input,CPUprice)
    #if there is no matches
    CPUresult=""
    if isinstance(recommend,str):
        CPUresult="no matches found"
    else:
        if recommend.empty:
            CPUresult="no matches found"
        else:
            CPUresult = recommend.iloc[0][["CpuName","Price"]].tolist()
            
    
    GPUinput = ""
    if input.split()[1] == "very":
        GPUinput="Design-Engineering, 3D Modeling, Simulation"
    else:
        GPUinput=input
        
    GPUrecommend = GPU_Recomm_System(GPUinput,GPUprice)
    if isinstance(GPUrecommend,str):
        GPUresult="no matches found"
    else:
        if GPUrecommend.empty:
            GPUresult="no matches found"
        else:
            GPUresult = GPU_Recomm_System(input,GPUprice).iloc[0][["GpuName","Price"]].tolist()
    
    result = {
        "CPU":CPUresult,
        "GPU": GPUresult,
        "MB": MBresult,
        "RAM":RAMresult,
        "HDD":HDDresult,
        "SSD":SSDresult,
        "Case":Caseresult,
        "PSU":PSUresult,
        "Cooler":Coolerresult
    }
    return jsonify(result),200
    

if __name__ == "__main__":
    app.run(debug=True)

    