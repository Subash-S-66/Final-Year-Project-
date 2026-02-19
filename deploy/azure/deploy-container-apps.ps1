param(
    [string]$SubscriptionId = "",
    [string]$ResourceGroup = "isl-collage-rg",
    [string]$Location = "eastus",
    [string]$AcrName = "islcollageacr",
    [string]$EnvironmentName = "isl-collage-env",
    [string]$BackendAppName = "isl-collage-backend",
    [string]$FrontendAppName = "isl-collage-frontend",
    [string]$ImageTag = "",
    [int]$BackendMinReplicas = 1,
    [int]$BackendMaxReplicas = 3,
    [int]$FrontendMinReplicas = 1,
    [int]$FrontendMaxReplicas = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ImageTag)) {
    $ImageTag = Get-Date -Format "yyyyMMddHHmmss"
}

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI (az) is not installed or not available in PATH."
}

if ($SubscriptionId) {
    az account set --subscription $SubscriptionId
}

function Test-ContainerAppExists {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Group
    )

    az containerapp show --name $Name --resource-group $Group --query name -o tsv 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

Write-Host "Ensuring Azure CLI extension: containerapp"
az extension add --name containerapp --upgrade --yes | Out-Null

Write-Host "Creating/ensuring resource group: $ResourceGroup"
az group create --name $ResourceGroup --location $Location | Out-Null

Write-Host "Creating/ensuring ACR: $AcrName"
az acr create `
    --resource-group $ResourceGroup `
    --name $AcrName `
    --sku Standard `
    --admin-enabled true | Out-Null

$acrLoginServer = az acr show --name $AcrName --resource-group $ResourceGroup --query loginServer -o tsv
$acrUser = az acr credential show --name $AcrName --query username -o tsv
$acrPass = az acr credential show --name $AcrName --query "passwords[0].value" -o tsv

Write-Host "Creating/ensuring Container Apps environment: $EnvironmentName"
az containerapp env create `
    --name $EnvironmentName `
    --resource-group $ResourceGroup `
    --location $Location | Out-Null

$backendImage = "$acrLoginServer/isl-backend:$ImageTag"
$frontendImage = "$acrLoginServer/isl-frontend:$ImageTag"

Write-Host "Building backend image in ACR: $backendImage"
az acr build `
    --registry $AcrName `
    --image $backendImage `
    --file backend/Dockerfile `
    . | Out-Null

if (Test-ContainerAppExists -Name $BackendAppName -Group $ResourceGroup) {
    Write-Host "Updating backend container app: $BackendAppName"
    az containerapp update `
        --name $BackendAppName `
        --resource-group $ResourceGroup `
        --image $backendImage `
        --set-env-vars ISL_USE_ONNX=true `
        --min-replicas $BackendMinReplicas `
        --max-replicas $BackendMaxReplicas | Out-Null
}
else {
    Write-Host "Creating backend container app: $BackendAppName"
    az containerapp create `
        --name $BackendAppName `
        --resource-group $ResourceGroup `
        --environment $EnvironmentName `
        --image $backendImage `
        --ingress external `
        --target-port 8000 `
        --transport auto `
        --cpu 1.0 `
        --memory 2.0Gi `
        --min-replicas $BackendMinReplicas `
        --max-replicas $BackendMaxReplicas `
        --registry-server $acrLoginServer `
        --registry-username $acrUser `
        --registry-password $acrPass `
        --env-vars ISL_USE_ONNX=true | Out-Null
}

$backendFqdn = az containerapp show `
    --name $BackendAppName `
    --resource-group $ResourceGroup `
    --query properties.configuration.ingress.fqdn `
    -o tsv

$frontendWsUrl = "wss://$backendFqdn/ws"

Write-Host "Building frontend image in ACR: $frontendImage"
Write-Host "Frontend VITE_WS_URL: $frontendWsUrl"
az acr build `
    --registry $AcrName `
    --image $frontendImage `
    --file frontend/Dockerfile `
    --build-arg "VITE_WS_URL=$frontendWsUrl" `
    . | Out-Null

if (Test-ContainerAppExists -Name $FrontendAppName -Group $ResourceGroup) {
    Write-Host "Updating frontend container app: $FrontendAppName"
    az containerapp update `
        --name $FrontendAppName `
        --resource-group $ResourceGroup `
        --image $frontendImage `
        --min-replicas $FrontendMinReplicas `
        --max-replicas $FrontendMaxReplicas | Out-Null
}
else {
    Write-Host "Creating frontend container app: $FrontendAppName"
    az containerapp create `
        --name $FrontendAppName `
        --resource-group $ResourceGroup `
        --environment $EnvironmentName `
        --image $frontendImage `
        --ingress external `
        --target-port 80 `
        --transport auto `
        --cpu 0.5 `
        --memory 1.0Gi `
        --min-replicas $FrontendMinReplicas `
        --max-replicas $FrontendMaxReplicas `
        --registry-server $acrLoginServer `
        --registry-username $acrUser `
        --registry-password $acrPass | Out-Null
}

$frontendFqdn = az containerapp show `
    --name $FrontendAppName `
    --resource-group $ResourceGroup `
    --query properties.configuration.ingress.fqdn `
    -o tsv

Write-Host ""
Write-Host "Deployment complete" -ForegroundColor Green
Write-Host "Frontend URL: https://$frontendFqdn"
Write-Host "Backend URL:  https://$backendFqdn"
Write-Host "Backend WS:   wss://$backendFqdn/ws"
