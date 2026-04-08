import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit         = 'origin/main' 
    self.download          = ['git://https://github.com/libxsmm/libxsmm.git']
    self.downloaddirnames  = ['libxsmm']
    self.includes          = ['libxsmm.h']
    self.liblist           = [['libxsmm.a']]
    self.functions         = ['libxsmm_init']
    self.precisions        = ['single', 'double']
    self.buildLanguages    = ['C', 'Cxx']
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.make = framework.require('config.packages.make', self)
    return

  def Install(self):
    conffile = os.path.join(self.packageDir, self.package + '.petscconf')
    with open(conffile, 'w') as f:
      f.write(self.installDir + '\n')
    if not self.installNeeded(conffile):
      return self.installDir

    import platform
    is_arm_mac = (platform.system() == 'Darwin' and platform.machine() == 'arm64')

    args  = 'PREFIX=' + self.installDir
    args += ' CC="'  + self.getCompiler('C')   + '"'
    args += ' CXX="' + self.getCompiler('Cxx') + '"'

    if is_arm_mac:
        # // changed: added STATIC=1 to ensure we get the .a file PETSc expects
        args += ' PLATFORM=1 JIT=1 STATIC=1'
    
    self.logPrintBox('Compiling LIBXSMM; this may take several minutes')
    
    make_cmd = 'cd ' + self.packageDir + ' && ' + self.make.make + ' -j8 ' + args + ' install'
    
    output, err, ret = config.package.Package.executeShellCommand(
      make_cmd,
      timeout = 600,
      log     = self.log
    )
    
    if ret:
      raise RuntimeError('Error building/installing LIBXSMM:\n' + output + err)

    self.postInstall(output + err, conffile)
    return self.installDir