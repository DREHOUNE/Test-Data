pipeline {
agent any
        environment {
        MYWORKDIR = '/Users/macbook/Desktop/jenkins'
        WORKSPACE = "${env.WORKSPACE}"
    }
stages {
        stage("Build"){
       steps{
         sh '''
           mvn clean install -DskipTests
           '''
         }
     }
         stage("Prepare env"){
       steps{
         sh '''
         cp $WORKSPACE/target/*.jar $MYWORKDIR && cp $WORKSPACE/submit.sh  $MYWORKDIR
         '''
         }
     }
          stage("Deploy"){
       steps{
         sh '''
          cd $MYWORKDIR && sh submit.sh
           '''
         }
     }
}
}