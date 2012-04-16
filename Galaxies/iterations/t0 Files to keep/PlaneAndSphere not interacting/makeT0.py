import sys
sys.dont_write_bytecode=True
from simulation import Simulation
from probabilityDistribution import ProbabilityDistribution
from particleDistribution import ParticleDistribution
from particleAgent import ParticleAgent

if  __name__  ==  '__main__':

  sim=Simulation('./0.xml')
  sim.initOutput()

  distrib=ParticleDistribution(4000, False, 0, 1)
  distribMass=ProbabilityDistribution('fixed', 0.00)
  distribPos=ProbabilityDistribution('linear', -1, 1)
  distribxVels=ProbabilityDistribution('fixed', -0.0)
  distribyVels=ProbabilityDistribution('fixed', -0.0)
  distribzVels=ProbabilityDistribution('fixed', -0.0)

  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribxVels,distribyVels,distribzVels) 
  sim.writeAgents(distrib.getParticleAgents())


  distrib=ParticleDistribution(1000, True, 100, 1)
  distribMass=ProbabilityDistribution('fixed', 0.01)
  distribPos=ProbabilityDistribution('circle', (0.0,0.0,-1.0), 0.2)
  distribxVels=ProbabilityDistribution('fixed', 0.0)
  distribyVels=ProbabilityDistribution('fixed', -0.0)
  distribzVels=ProbabilityDistribution('fixed', -0.0)

  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribxVels,distribyVels,distribzVels) 
  sim.writeAgents(distrib.getParticleAgents())

  sim.closeOutput()
