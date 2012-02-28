import  random

class  ParticleAgent:
    def  __init__(self,  particleId,  xPos,  yPos,  zPos,  xVel,  yVel,  zVel,  mass,  isDark):
        self.particleId=particleId
        self.xPos=xPos
        self.yPos=yPos
        self.zPos=zPos
        self.xVel=xVel
        self.yVel=yVel
        self.zVel=zVel
        self.mass=mass
        self.isDark=isDark

    def  writeAgent(self):
        outStr=""
        outStr+='<xagent>'
        outStr+='<name>Particle</name>'
        outStr+='<id>'
        outStr+=str(self.particleId)
        outStr+='<mass>'
        outStr+=str(self.mass)
        outStr+='</mass>'
        outStr+='<isDark>0</isDark>'
        outStr+='<x>'
        outStr+=str(self.xPos)
        outStr+='</x>'
        outStr+='<y>'
        outStr+=str(self.yPos)
        outStr+='</y>'
        outStr+='<z>'
        outStr+=str(self.zPos)
        outStr+='</z>'
        outStr+='<xVel>'
        outStr+=str(self.xVel)
        outStr+='</xVel>'
        outStr+='<yVel>'
        outStr+=str(self.yVel)
        outStr+='</yVel>'
        outStr+='<zVel>'
        outStr+=str(self.zVel)
        outStr+='</zVel>'
        outStr+='</xagent>\r\n'
        return  outStr

class  Simulation:
    def  __init__(self,  filename):
        self.filename=filename

    def  initOutput(self):
        self.outputFile=open(self.filename,  'w')
        self.outputFile.write('<states>\r\n<itno>0</itno>\r\n')

    def  closeOutput(self):
        self.outputFile.write('</states>')
        self.outputFile.close()

    def  writeAgents(self,  ls):
        for  agent  in  ls:
            agentXML=agent.writeAgent()
            self.outputFile.write(agentXML)
            

class  ProbabilityDistribution:
    def  __init__(self,  distribType,  val1=None,  val2=None):
        if(distribType  is  ('gaussian'  or  'normal')):
            self.distribType='gaussian'

            if((val1  or  val2)  is  None):
                print  "ERROR:  Must  define  a  mu  and  sigma  value  for  a  Gaussian  distribution"
                exit()

            else:
                self.mu=val1
                self.sigma=val2

        elif(distribType  is  ('fixed'  or  'set'  or  'static'))  :
            self.distribType='fixed'

            if(val1  is  None):
                print  "ERROR:  Must  define  a  set  value  for  a  fixed  distribution"
                exit()

            else:
                self.fixedVal=val1

        else:
            self.distribType='random'

            if((val1  or  val2)  is  None):
                print  "ERROR:  Must  define  a  Min  and  Max  value  for  a  random  distribution"
                exit()

            else:
                self.minVal=val1
                self.maxVal=val2

    def  getItem(self):
        if(self.distribType  is  'gaussian'):
            return  random.gauss(self.mu,  self.sigma)

        elif(self.distribType  is  'random'):
            return  random.uniform(self.minVal,  self.maxVal)

        elif(self.distribType  is  'fixed'):
            return  self.fixedVal


class  ParticleDistribution:
    def  __init__(self,  numAgents,  usingZAxis):
        self.numAgents=numAgents
        self.usingZAxis=usingZAxis
        self.particles={}
        self.massesSet=False
        self.positionsSet=False
        self.velocitiesSet=False

        for  counter  in  range  (0,  self.numAgents):
            self.particles[counter]=[0,False,(0,0,0),(0,0,0)]

    def  setMasses(self,  massDistribution):
        for  count  in  range  (0,  self.numAgents):
            self.particles[count][0]=massDistribution.getItem()

            #darkMatter
            self.particles[count][1]=False
            self.massesSet=True
    
    def  setPositions(self,  xDistrib,  yDistrib=None,  zDistrib=None):
        #If  not  set,  use  same  distribution  for  all  dimensions
        if((yDistrib  and  zDistrib)  is  None):
            yDistrib=xDistrib
            zDistrib=xDistrib

        for  i  in  range  (0,  self.numAgents):
            self.particles[i][3]=(xDistrib.getItem(),  yDistrib.getItem(),  zDistrib.getItem())

        self.positionsSet=True
    
    def  setVelocities(self,  xVelDistrib,  yVelDistrib=None,  zVelDistrib=None):

        #If  not  set,  use  same  distribution  for  all  dimensions
        if((yVelDistrib  and  zVelDistrib)  is  None):
            yVelDistrib=xVelDistrib
            zVelDistrib=xVelDistrib

        for  i  in  range  (0,  self.numAgents):
            self.particles[i][3]=(xVelDistrib.getItem(),  yVelDistrib.getItem(),  zVelDistrib.getItem())

        self.velocitiesSet=True

    def  makeAgentsFromFile(self,  fileloc):
            agentsDict={}

            #File  loading  code
            scaleFactor=1
            velScaleFactor=1
            massScale=1
            inFile=open(fileloc,  'r')

            count=0
            with  inFile  as  f:
                content=f.readlines()

            for  line  in  content:
                stripStr=line.split()
                massPosVel=[0,0,0,0,0,0,0]
                massPosVel[0]=massScale*float(stripStr[0])

            for  index  in  range(1,4):
                massPosVel[index]=scaleFactor*float(stripStr[index])

            for  index  in  range(4,7):
                massPosVel[index]=velScaleFactor*float(stripStr[index])

            count+=1

            agentsDict[count]=massPosVel

            for  item  in  agentsDict:
                ls=agentsDict[item]
                newAgent=ParticleAgent(item,  ls[1],  ls[2],  ls[3],  ls[4],  ls[5],  ls[6],  ls[0],  False)
                agentXML=newAgent.writeAgent()
                self.outputFile.write(agentXML)

    def  getParticleAgents(self):

        particleAgents=[]
        if(not(self.massesSet  and  self.positionsSet  and  self.velocitiesSet)):
            print  "ERROR:  Can't  write  particle  positions  until  masses,  positions  and  velocities  have  been  set"
    
        else:
            for  key,  val  in  self.particles.iteritems():
	mass=val[0]
	isDark=val[1]
                xPos=val[2][0]
                yPos=val[2][1]
                zPos=val[2][2]
                xVel=val[3][0]
                yVel=val[3][1]
                zVel=val[3][2]
                thisParticle=ParticleAgent(key,  xPos,  yPos,  zPos,  xVel,  yVel,  zVel,  mass,  isDark)
	particleAgents.append(thisParticle)

        return  particleAgents

if  __name__  ==  '__main__':
    numAgents=1000
    maxMass=0.05
    maxVel=0.7
    maxPos=0.3
    minPos=0
    randomMass=False
    randomVel=True
    usingZAxis=True

    simulation=Simulation('./0.xml')
    simulation.initOutput()

    distribution1=ParticleDistribution(2,  True)
    distrib1Masses=ProbabilityDistribution('fixed',  0.05)
    distrib1positions=ProbabilityDistribution('random',  0.3,  0)
    distrib1velocities=ProbabilityDistribution('random',  0.7,  0)

    distribution1.setMasses(distrib1Masses)
    distribution1.setPositions(distrib1positions)
    #distribution1.testFn()
    distribution1.setVelocities(distrib1velocities)

    distrib1Particles=distribution1.getParticleAgents()
    simulation.writeAgents(distrib1Particles)

    #distrib=Simulation(outFile,  5000,  0.05,  0.7,  0.7,  .7,  -1.25,  False,  randomVel,  usingZAxis)
    #sim.initOutput()

    #sim.makeAgentsFromFile('tab8096.txt')
    #sim.makeAgents()
    #distrib.makeAgents()

    simulation.closeOutput()

