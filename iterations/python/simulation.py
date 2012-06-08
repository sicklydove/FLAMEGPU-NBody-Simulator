import sys
sys.dont_write_bytecode=True
from particleDistribution import *

class Simulation:
  def  __init__(self, filename):
    self.filename=filename

  def initOutput(self):
    self.outputFile=open(self.filename, 'w')
    self.outputFile.write('<states>\r\n<itno>0</itno>\r\n')

  def closeOutput(self):
    self.outputFile.write('</states>')
    self.outputFile.close()

  def writeAgents(self, agentList):
    for agent in agentList:
      agentXML=agent.getAgentXML()
      self.outputFile.write(agentXML)
