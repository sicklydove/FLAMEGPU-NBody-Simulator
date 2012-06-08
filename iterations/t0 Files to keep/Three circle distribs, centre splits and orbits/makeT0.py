import sys
sys.dont_write_bytecode=True
from simulation import Simulation
from probabilityDistribution import ProbabilityDistribution
from particleDistribution import ParticleDistribution
from particleAgent import ParticleAgent

if  __name__  ==  '__main__':

  sim=Simulation('./0.xml')
  sim.initOutput()

  distrib=ParticleDistribution(1000, False, 0, 1)
  distribMass=ProbabilityDistribution('fixed', 0.035)
  distribPos=ProbabilityDistribution('circle', (1.2,1.2,0.0), 0.2)
  distribxVels=ProbabilityDistribution('fixed', -4.0)
  distribyVels=ProbabilityDistribution('fixed', -0.0)
  distribzVels=ProbabilityDistribution('fixed', -0.0)

  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribxVels,distribyVels,distribzVels) 
  sim.writeAgents(distrib.getParticleAgents())


  distrib=ParticleDistribution(500, False, 100, 1)
  distribMass=ProbabilityDistribution('fixed', 0.01)
  distribPos=ProbabilityDistribution('circle', (0.6,0.6,0.0), 0.2)
  distribxVels=ProbabilityDistribution('fixed', 0.0)
  distribyVels=ProbabilityDistribution('fixed', -0.0)
  distribzVels=ProbabilityDistribution('fixed', -0.0)

  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribxVels,distribyVels,distribzVels) 
  sim.writeAgents(distrib.getParticleAgents())


  distrib=ParticleDistribution(1000, False, 0, 1)
  distribMass=ProbabilityDistribution('fixed', 0.035)
  distribPos=ProbabilityDistribution('circle', (0,0,0.0), 0.2)
  distribxVels=ProbabilityDistribution('fixed', 4.0)
  distribyVels=ProbabilityDistribution('fixed', -0.0)
  distribzVels=ProbabilityDistribution('fixed', -0.0)

  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribxVels,distribyVels,distribzVels) 
  sim.writeAgents(distrib.getParticleAgents())

# distrib=ParticleDistribution(500, True, 100, 2)
# distribMass=ProbabilityDistribution('fixed', 0.05)
# distribPos=ProbabilityDistribution('circle', (-0.5,-0.3,-0), 0.1)
# distribVels=ProbabilityDistribution('linear', -0.5,0.5)
# distrib.setMasses(distribMass)
# distrib.setPositions(distribPos)
# distrib.setVelocities(distribVels) 
# sim.writeAgents(distrib.getParticleAgents())

# distrib=ParticleDistribution(500, True, 100, 2)
# distribMass=ProbabilityDistribution('fixed', 0.05)
# distribPos=ProbabilityDistribution('circle', (1,0,1), 0.1)
# distribVels=ProbabilityDistribution('linear', -0.5,0.5)
# distrib.setMasses(distribMass)
# distrib.setPositions(distribPos)
# distrib.setVelocities(distribVels) 
# sim.writeAgents(distrib.getParticleAgents())

# distrib=ParticleDistribution(500, True, 100, 2)
# distribMass=ProbabilityDistribution('fixed', 0.05)
# distribPos=ProbabilityDistribution('circle', (1,0.5,.3), 0.1)
# distribVels=ProbabilityDistribution('linear', -0.5,0.5)
# distrib.setMasses(distribMass)
# distrib.setPositions(distribPos)
# distrib.setVelocities(distribVels) 
# sim.writeAgents(distrib.getParticleAgents())
  sim.closeOutput()
