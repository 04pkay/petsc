import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version           = '1.17'
    self.gitcommit         = self.version
    self.download          = ['git://https://github.com/libxsmm/libxsmm.git',
                              'https://github.com/libxsmm/libxsmm/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames  = ['libxsmm']
    self.includes          = ['libxsmm.h']
    self.liblist           = [['libxsmm.a', 'libxsmmext.a']]
    self.functions         = ['libxsmm_init']
    self.precisions        = ['single', 'double']
    self.buildLanguages    = ['C', 'Cxx']
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.make = framework.require('config.packages.make', self)
    return

  def Install(self):
    # Write a conffile so PETSc can skip reinstall when nothing changed
    conffile = os.path.join(self.packageDir, self.package + '.petscconf')
    with open(conffile, 'w') as f:
      f.write(self.installDir + '\n')
    if not self.installNeeded(conffile):
      return self.installDir

    # LIBXSMM's Makefile honours PREFIX, CC, CXX and installs via 'make install'
    args  = 'PREFIX=' + self.installDir
    args += ' CC="'  + self.getCompiler('C')   + '"'
    args += ' CXX="' + self.getCompiler('Cxx') + '"'
    #args += ' PLATFORM=1 ARCH=generic'

    self.logPrintBox('Compiling LIBXSMM; this may take several minutes')
    output, err, ret = config.package.Package.executeShellCommand(
      'cd ' + self.packageDir + ' && ' + self.make.make + ' ' + args + ' install',
      timeout = 300,
      log     = self.log
    )
    if ret:
      raise RuntimeError('Error building/installing LIBXSMM:\n' + output + err)

    self.postInstall(output + err, conffile)
    return self.installDir