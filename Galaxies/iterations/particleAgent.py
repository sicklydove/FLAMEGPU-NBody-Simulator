import random
import sys
sys.dont_write_bytecode=True
from particleDistribution import *
from math import *

class  ParticleAgent:
  def  __init__(self, particleId, xPos,  yPos,  zPos,  xVel,  yVel,  zVel,  mass,  isDark):
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
    rand = random.random()
    if (rand>0.5):
      rand = 1
    else:
      rand=0

    outStr=""
    outStr+='<xagent>'
    outStr+='<name>Particle</name>'
    outStr+='<id>'
    outStr+=str(self.particleId)
    outStr+='</id>'
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
    outStr+='<initialOffset>'
#    outStr+=str(0)
    outStr+=str(rand)
    outStr+='</initialOffset>'
    outStr+='</xagent>\r\n'
    return  outStr

class  Simulation:
  def  __init__(self,  filename):
    self.filename=filename

  def  initOutput(self):
    self.outputFile=open(self.filename,  'w')
    self.outputFile.write('<states>\r\n<itno>0</itno>\r\n')
    self.outputFile.write('<xagent><name>simulationVarsAgent</name><iterationNum>0</iterationNum></xagent>\r\n')

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

      if((val1 or val2)  is  None):
        print  "ERROR:  Must  define  a  mu  and  sigma  value  for  a  Gaussian  distribution"
        exit()

      else:
        self.mu=val1
        self.sigma=val2

    elif(distribType is ('fixed'  or  'set'  or  'static')):
      self.distribType='fixed'

      if(val1  is  None):
        print  "ERROR:  Must  define  a  set  value  for  a  fixed  distribution"
        exit()

      else:
        self.fixedVal=val1

    elif(distribType is 'circle'):
      self.distribType='circle'
      self.centre=val1
      self.radius=val2

    else:
      self.distribType='linear'

      if((val1  or  val2)  is  None):
        print  "ERROR:  Must  define  a  Min  and  Max  value  for  a  random  distribution"
        exit()

      else:
        self.minVal=val1
        self.maxVal=val2

  def  getItem(self):
    if(self.distribType  is  'gaussian'):
      return  random.gauss(self.mu,  self.sigma)

    elif(self.distribType  is  'linear'):
      return  random.uniform(self.minVal,  self.maxVal)

    elif(self.distribType  is  'fixed'):
      return  self.fixedVal

    elif(self.distribType is 'circle'):
      randRadius=random.uniform(0, self.radius)
      randxPos=random.uniform(self.centre[0]-randRadius, self.centre[0]+randRadius)

      xPosSq=(randxPos-self.centre[0])**2
      radiusSq=randRadius**2

      maxYMinusBSq=radiusSq-xPosSq

      maxY=quadratic(1, 0-(2*self.centre[1]), ((self.centre[1]**2)-maxYMinusBSq))[0]

      randyPos=random.uniform(self.centre[1], maxY)

      randyPosSq=(randyPos-self.centre[1])**2
     
      if(random.random()>0.5):
        randyPos=self.centre[1]+(self.centre[1]-randyPos)

      zMinusCSq=radiusSq-xPosSq-randyPosSq

      zMinusC=sqrt(zMinusCSq)

      if(random.random()>0.5):
        zPos=zMinusC+self.centre[2]
      else:
        zPos=self.centre[2]-zMinusC
      return (randxPos,randyPos,zPos)
    
  def getType(self):
    return self.distribType

#stackoverflow code
def quadratic(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
      return []
    elif discriminant == 0:
      return [-b / (2*a)]
    else:
      root = sqrt(discriminant)
      ls = [(-b + root) / (2*a), (-b - root) / (2*a)]
      return filter(lambda x: x>0, ls)

