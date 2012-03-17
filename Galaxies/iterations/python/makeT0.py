import sys
sys.dont_write_bytecode=True
#import particleAgent
from simulation import Simulation 
from probabilityDistribution import ProbabilityDistribution
from particleDistribution import ParticleDistribution
import particleAgent

if  __name__  ==  '__main__':

  simulation=Simulation('./0.xml')
  simulation.initOutput()

  distrib=ParticleDistribution(2000, True)
  distribMass=ProbabilityDistribution('fixed', 0.05)
  distribPos=ProbabilityDistribution('circle',(1,1,0.5), 0.1)
  distribVels=ProbabilityDistribution('linear', -0.5,0.5)
  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribVels) 
  simulation.writeAgents(distrib.getParticleAgents())

  distrib=ParticleDistribution(500, True)
  distribMass=ProbabilityDistribution('fixed', 0.001)
  distribPos=ProbabilityDistribution('circle',(0,0,0), 3.5)
  distribVels=ProbabilityDistribution('linear', -0.5,0.5)
  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribVels) 
  simulation.writeAgents(distrib.getParticleAgents())

  distrib=ParticleDistribution(500, True)
  distribMass=ProbabilityDistribution('fixed', 0.05)
  distribPos=ProbabilityDistribution('circle',(-1,-1,1.32), 0.1)
  distribVels=ProbabilityDistribution('linear', -0.5,0.5)
  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribVels) 
  simulation.writeAgents(distrib.getParticleAgents())

  distrib=ParticleDistribution(500, True)
  distribMass=ProbabilityDistribution('fixed', 0.05)
  distribPos=ProbabilityDistribution('circle',(-0.3,0.9,-0.3), 0.1)
  distribVels=ProbabilityDistribution('linear', -0.5,0.5)
  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribVels) 
  simulation.writeAgents(distrib.getParticleAgents())

  simulation.closeOutput()
