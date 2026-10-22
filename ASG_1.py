print("Developed by: Nikunj Tangle")
print("Submitted to: Vaishali Ingale Mam")


companies={
    "TCS": {"cgpa":6.0, "placement": "3 LPA", "branches":["CSE","IT","ENTC","MECH"]},
    "INFOSYS": {"cgpa":7.0, "placement": "5 LPA", "branches":["CSE","IT","ENTC"]},
    "MICROSOFT": {"cgpa":8.5, "placement": "18 LPA", "branches":["CSE","IT"]},
    "GOOGLE": {"cgpa":9.0, "placement": "25 LPA", "branches":["CSE","IT"]}
}

BRANCHES = ["CSE", "IT", "ECE", "EEE", "ME", "CE"]
MENU='''
1. View Companies
2. Check Eligibility
3. Resume Tips
4. Interview Tips
5. Exit
'''


def sensor():
    return input("You:").strip()

def actuator(response):
    print("Bot:", response, "\n")


class SimpleReflexAgent:
    def response(self,text):
        if text=="1":
            return "Companies"+",".join(companies)
        elif text=="2":
            return "Sorry, I dont have any memory"
        elif text=="3":
            return "Keep it one page, and include projects and experience"
        elif text=="4":
            return "Be confident and believe in yourself"
        elif text=="5":
            return "Happy Placements!!"
        else:
            return "Please enter a value between 1-5"
        


class ModelBasedAgent:
    def __init__(self):
        self.cgpa=None
        self.branch=None
        
    def response(self,text):
        if text.lower().startswith("cgpa"):
            self.cgpa=float(text.split()[-1])
            return f"CGPA saved{self.cgpa}"
        if text.upper() in BRANCHES:
            self.branch=text.upper()
            return f"Branch: {self.branch}"
        
        for name,info in companies.items():
            if name.lower()==text.lower():
                return f"{name}-> Min CGPA: {info["cgpa"]}, Package: {info["placement"]} , Branches:{info["branches"]}"
            
            
        if text=="1":
            return "Companies"+",".join(companies)
        elif text=="2":
            if self.cgpa is None or self.branch is None:
                return "Enter cgpa and branch"
            eligible=[]
            for name, info in companies.items():
                if self.cgpa>=info["cgpa"]:
                    if self.branch in info["branches"]:
                        company_details=f"{name}({info["placement"]})"
                        eligible.append(company_details)
                        
            if eligible:
                return f"The Companies you are eligible in are:"+",".join(eligible)
                
        elif text=="3":
            return "Keep it one page, and include projects and experience"
        elif text=="4":
            return "Be confident and believe in yourself"
        elif text=="5":
            return "Happy Placements!!"
        else:
            return "Please enter a value between 1-5"
            


print("Simple vs Model Based Agent")
print("Choose 1. Simple Reflex Agent or 2. Model Based Agent")
choice=int(input("Enter your choice between 1 or 2"))

if choice==1:
    agent=SimpleReflexAgent()
    print("Simple Reflex Agent")
    
else:
    agent=ModelBasedAgent()
    print("Model Based Agent")
    
print(MENU, "\n")

while True:
    text=sensor()
    reply=agent.response(text)
    actuator(reply)
    if text=="5":
        break
    
    
    
