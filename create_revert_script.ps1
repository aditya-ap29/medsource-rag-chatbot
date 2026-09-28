$content = @'
Write-Host "Reverting to the pre-reembedding backup..."

if (Test-Path "data\vector_store_backup_pre_reembed") {
    if (Test-Path "data\vector_store") {
        Remove-Item -Recurse -Force "data\vector_store"
    }
    Copy-Item -Recurse "data\vector_store_backup_pre_reembed" "data\vector_store"
    Write-Host "Restored data\vector_store from backup."
} else {
    Write-Host "ERROR: backup folder not found at data\vector_store_backup_pre_reembed"
    exit 1
}

'@
Set-Content -Path 'revert_embedding.ps1' -Value $content -Encoding utf8
Write-Host 'revert_embedding.ps1 has been created.'$content = @'
Write-Host "Reverting to the pre-reembedding backup..."

if (Test-Path "data\vector_store_backup_pre_reembed") {
    if (Test-Path "data\vector_store") {
        Remove-Item -Recurse -Force "data\vector_store"
    }
    Copy-Item -Recurse "data\vector_store_backup_pre_reembed" "data\vector_store"
    Write-Host "Restored data\vector_store from backup."
} else {
    Write-Host "ERROR: backup folder not found at data\vector_store_backup_pre_reembed"
    exit 1
}

'@
Set-Content -Path 'revert_embedding.ps1' -Value $content -Encoding utf8
Write-Host 'revert_embedding.ps1 has been created.'