import sys
sys.dont_write_bytecode=True
from particleDistribution import *
from math import *

class Simulation:
  def  __init__(self, filename, numParticleGroups):
    self.filename=filename
    self.numParticleGroups=numParticleGroups

  def initOutput(self):
    self.outputFile=open(self.filename,  'w')
    self.outputFile.write('<states>\r\n<itno>0</itno>\r\n')
    self.outputFile.write('<xagent><name>simulationVarsAgent</name><iterationNum>0</iterationNum></xagent>\r\n')

  def closeOutput(self):
    self.outputFile.write('</states>')
    self.outputFile.close()

  def writeAgents(self, ls):
    for  agent  in  ls:
      agentXML=agent.writeAgent()
      self.outputFile.write(agentXML)
