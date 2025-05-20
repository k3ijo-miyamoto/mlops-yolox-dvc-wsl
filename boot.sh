#!/bin/bash
set -e

NAMESPACE=mlops-test

kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

eval $(minikube docker-env)

docker build --no-cache -t mlops-yolox:latest .

kubectl apply -f deployment.yaml -n $NAMESPACE
kubectl apply -f service.yaml -n $NAMESPACE
