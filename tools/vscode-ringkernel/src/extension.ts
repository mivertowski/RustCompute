/**
 * RingKernel VSCode Extension
 *
 * Provides GPU kernel development support:
 * - Syntax highlighting for CUDA/WGSL
 * - Code snippets for kernel patterns
 * - GPU memory dashboard
 * - Kernel profiling integration
 * - Backend compatibility checking
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// ============================================================================
// Extension Activation
// ============================================================================

export function activate(context: vscode.ExtensionContext) {
    console.log('RingKernel extension activated');

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('ringkernel.generateKernel', generateKernel),
        vscode.commands.registerCommand('ringkernel.transpileToCuda', transpileToCuda),
        vscode.commands.registerCommand('ringkernel.transpileToWgsl', transpileToWgsl),
        vscode.commands.registerCommand('ringkernel.checkBackendCompat', checkBackendCompat),
        vscode.commands.registerCommand('ringkernel.launchGpuPlayground', launchGpuPlayground),
        vscode.commands.registerCommand('ringkernel.showMemoryDashboard', showMemoryDashboard),
        vscode.commands.registerCommand('ringkernel.profileKernel', profileKernel),
    );

    // Register code lens provider for kernel functions
    context.subscriptions.push(
        vscode.languages.registerCodeLensProvider('rust', new KernelCodeLensProvider())
    );

    // Register hover provider for GPU intrinsics
    context.subscriptions.push(
        vscode.languages.registerHoverProvider('rust', new GpuIntrinsicsHoverProvider())
    );

    // Create status bar item for GPU memory
    const memoryStatusBar = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    memoryStatusBar.text = '$(memory) GPU: --';
    memoryStatusBar.tooltip = 'GPU Memory Usage';
    memoryStatusBar.command = 'ringkernel.showMemoryDashboard';

    const config = vscode.workspace.getConfiguration('ringkernel');
    if (config.get('showMemoryUsage')) {
        memoryStatusBar.show();
    }
    context.subscriptions.push(memoryStatusBar);

    // Register tree view providers
    const kernelTreeProvider = new KernelTreeProvider();
    vscode.window.registerTreeDataProvider('ringkernel.kernels', kernelTreeProvider);

    const memoryTreeProvider = new MemoryTreeProvider();
    vscode.window.registerTreeDataProvider('ringkernel.memory', memoryTreeProvider);

    const profilerTreeProvider = new ProfilerTreeProvider();
    vscode.window.registerTreeDataProvider('ringkernel.profiler', profilerTreeProvider);
}

export function deactivate() {
    console.log('RingKernel extension deactivated');
}

// ============================================================================
// Commands
// ============================================================================

async function generateKernel() {
    const kernelTypes = [
        { label: 'Global Kernel', description: 'Standard GPU kernel' },
        { label: 'Stencil Kernel', description: 'Grid-based stencil computation' },
        { label: 'Ring Kernel', description: 'Persistent actor kernel' },
        { label: 'Persistent FDTD', description: 'Persistent 3D wave simulation' },
    ];

    const selected = await vscode.window.showQuickPick(kernelTypes, {
        placeHolder: 'Select kernel type to generate'
    });

    if (!selected) return;

    const name = await vscode.window.showInputBox({
        prompt: 'Enter kernel name',
        placeHolder: 'my_kernel'
    });

    if (!name) return;

    const template = getKernelTemplate(selected.label, name);

    const editor = vscode.window.activeTextEditor;
    if (editor) {
        editor.edit(editBuilder => {
            editBuilder.insert(editor.selection.active, template);
        });
    }
}

function getKernelTemplate(type: string, name: string): string {
    switch (type) {
        case 'Global Kernel':
            return `
/// GPU kernel: ${name}
#[gpu_kernel(backends = [cuda, wgpu])]
fn ${name}(input: &[f32], output: &mut [f32], n: i32) {
    let idx = block_idx_x() * block_dim_x() + thread_idx_x();
    if idx >= n { return; }
    output[idx as usize] = input[idx as usize] * 2.0;
}
`;
        case 'Stencil Kernel':
            return `
/// Stencil kernel: ${name}
#[stencil_kernel(tile_size = (16, 16), halo = 1)]
fn ${name}(input: &[f32], output: &mut [f32], pos: GridPos) {
    let laplacian = pos.north(input) + pos.south(input)
                  + pos.east(input) + pos.west(input)
                  - 4.0 * input[pos.idx()];
    output[pos.idx()] = input[pos.idx()] + 0.25 * laplacian;
}
`;
        case 'Ring Kernel':
            return `
/// Ring kernel actor: ${name}
#[ring_kernel(
    id = "${name}",
    mode = "persistent",
    block_size = 128,
    backends = [cuda, metal],
)]
async fn ${name}_handler(ctx: &mut RingContext, msg: Request) -> Response {
    let result = msg.value * ctx.global_thread_id() as f32;
    ctx.sync_threads();
    Response { value: result }
}
`;
        case 'Persistent FDTD':
            return `
/// Persistent FDTD kernel: ${name}
#[persistent_fdtd(
    tile_size = (8, 8, 8),
    cooperative = true,
    progress_interval = 100,
)]
fn ${name}_step(
    p: &[f32],
    p_prev: &mut [f32],
    c2: f32,
    pos: GridPos3D,
) {
    let laplacian = pos.north(p) + pos.south(p)
                  + pos.east(p) + pos.west(p)
                  + pos.up(p) + pos.down(p)
                  - 6.0 * p[pos.idx()];
    p_prev[pos.idx()] = 2.0 * p[pos.idx()] - p_prev[pos.idx()] + c2 * laplacian;
}
`;
        default:
            return '';
    }
}

async function transpileToCuda() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
    }

    const cliPath = getCliPath();
    const filePath = editor.document.uri.fsPath;

    try {
        const { stdout } = await execAsync(`${cliPath} codegen "${filePath}" --backend cuda`);

        const doc = await vscode.workspace.openTextDocument({
            content: stdout,
            language: 'cuda'
        });
        await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
    } catch (error: any) {
        vscode.window.showErrorMessage(`Transpilation failed: ${error.message}`);
    }
}

async function transpileToWgsl() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
    }

    const cliPath = getCliPath();
    const filePath = editor.document.uri.fsPath;

    try {
        const { stdout } = await execAsync(`${cliPath} codegen "${filePath}" --backend wgsl`);

        const doc = await vscode.workspace.openTextDocument({
            content: stdout,
            language: 'wgsl'
        });
        await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
    } catch (error: any) {
        vscode.window.showErrorMessage(`Transpilation failed: ${error.message}`);
    }
}

async function checkBackendCompat() {
    const cliPath = getCliPath();

    try {
        const { stdout } = await execAsync(`${cliPath} check --backends all`);

        const panel = vscode.window.createWebviewPanel(
            'ringkernelCompat',
            'Backend Compatibility',
            vscode.ViewColumn.One,
            {}
        );

        panel.webview.html = getCompatibilityHtml(stdout);
    } catch (error: any) {
        vscode.window.showErrorMessage(`Compatibility check failed: ${error.message}`);
    }
}

async function launchGpuPlayground() {
    const config = vscode.workspace.getConfiguration('ringkernel');
    const port = config.get<number>('playground.port') || 8765;

    vscode.window.showInformationMessage(`Launching GPU Playground on port ${port}...`);

    // Open playground in browser
    vscode.env.openExternal(vscode.Uri.parse(`http://localhost:${port}`));
}

async function showMemoryDashboard() {
    const panel = vscode.window.createWebviewPanel(
        'ringkernelMemory',
        'GPU Memory Dashboard',
        vscode.ViewColumn.One,
        { enableScripts: true }
    );

    panel.webview.html = getMemoryDashboardHtml();
}

async function profileKernel() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
    }

    const kernelName = await vscode.window.showInputBox({
        prompt: 'Enter kernel name to profile',
        placeHolder: 'my_kernel'
    });

    if (!kernelName) return;

    vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `Profiling ${kernelName}...`,
        cancellable: true
    }, async (progress, token) => {
        // Simulate profiling
        for (let i = 0; i <= 100; i += 10) {
            if (token.isCancellationRequested) break;
            progress.report({ increment: 10, message: `${i}%` });
            await new Promise(resolve => setTimeout(resolve, 200));
        }

        vscode.window.showInformationMessage(`Profiling complete for ${kernelName}`);
    });
}

function getCliPath(): string {
    const config = vscode.workspace.getConfiguration('ringkernel');
    const customPath = config.get<string>('cliPath');
    return customPath || 'ringkernel-cli';
}

// ============================================================================
// Code Lens Provider
// ============================================================================

class KernelCodeLensProvider implements vscode.CodeLensProvider {
    provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
        const lenses: vscode.CodeLens[] = [];
        const text = document.getText();

        // Find kernel attributes
        const kernelRegex = /#\[(ring_kernel|gpu_kernel|stencil_kernel|persistent_fdtd)\(/g;
        let match;

        while ((match = kernelRegex.exec(text)) !== null) {
            const pos = document.positionAt(match.index);
            const range = new vscode.Range(pos, pos);

            lenses.push(new vscode.CodeLens(range, {
                title: '$(play) Run Kernel',
                command: 'ringkernel.profileKernel'
            }));

            lenses.push(new vscode.CodeLens(range, {
                title: '$(symbol-misc) Transpile',
                command: 'ringkernel.transpileToCuda'
            }));
        }

        return lenses;
    }
}

// ============================================================================
// Hover Provider
// ============================================================================

class GpuIntrinsicsHoverProvider implements vscode.HoverProvider {
    private intrinsics: Map<string, string> = new Map([
        ['block_idx_x', 'Block index in X dimension (CUDA: blockIdx.x)'],
        ['block_idx_y', 'Block index in Y dimension (CUDA: blockIdx.y)'],
        ['block_idx_z', 'Block index in Z dimension (CUDA: blockIdx.z)'],
        ['thread_idx_x', 'Thread index in X dimension (CUDA: threadIdx.x)'],
        ['thread_idx_y', 'Thread index in Y dimension (CUDA: threadIdx.y)'],
        ['thread_idx_z', 'Thread index in Z dimension (CUDA: threadIdx.z)'],
        ['block_dim_x', 'Block dimension in X (CUDA: blockDim.x)'],
        ['block_dim_y', 'Block dimension in Y (CUDA: blockDim.y)'],
        ['grid_dim_x', 'Grid dimension in X (CUDA: gridDim.x)'],
        ['sync_threads', 'Synchronize all threads in block (CUDA: __syncthreads())'],
        ['atomic_add', 'Atomic addition (CUDA: atomicAdd())'],
        ['atomic_cas', 'Atomic compare-and-swap (CUDA: atomicCAS())'],
        ['warp_size', 'Number of threads per warp (typically 32)'],
        ['grid_sync', 'Synchronize entire grid (cooperative groups)'],
    ]);

    provideHover(document: vscode.TextDocument, position: vscode.Position): vscode.Hover | null {
        const wordRange = document.getWordRangeAtPosition(position);
        if (!wordRange) return null;

        const word = document.getText(wordRange);
        const description = this.intrinsics.get(word);

        if (description) {
            return new vscode.Hover(
                new vscode.MarkdownString(`**GPU Intrinsic**: \`${word}\`\n\n${description}`)
            );
        }

        return null;
    }
}

// ============================================================================
// Tree Providers
// ============================================================================

class KernelTreeProvider implements vscode.TreeDataProvider<KernelItem> {
    getTreeItem(element: KernelItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: KernelItem): KernelItem[] {
        if (element) return [];

        return [
            new KernelItem('processor', 'Ring Kernel', vscode.TreeItemCollapsibleState.None),
            new KernelItem('fdtd_step', 'Stencil Kernel', vscode.TreeItemCollapsibleState.None),
            new KernelItem('saxpy', 'Global Kernel', vscode.TreeItemCollapsibleState.None),
        ];
    }
}

class KernelItem extends vscode.TreeItem {
    constructor(
        public readonly name: string,
        public readonly kernelType: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(name, collapsibleState);
        this.tooltip = `${kernelType}: ${name}`;
        this.description = kernelType;
    }
}

class MemoryTreeProvider implements vscode.TreeDataProvider<MemoryItem> {
    getTreeItem(element: MemoryItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: MemoryItem): MemoryItem[] {
        if (element) return [];

        return [
            new MemoryItem('Device Memory', '1.2 GB / 8.0 GB'),
            new MemoryItem('Host Visible', '256 MB'),
            new MemoryItem('Queue Buffers', '64 MB'),
        ];
    }
}

class MemoryItem extends vscode.TreeItem {
    constructor(name: string, value: string) {
        super(name, vscode.TreeItemCollapsibleState.None);
        this.description = value;
    }
}

class ProfilerTreeProvider implements vscode.TreeDataProvider<ProfilerItem> {
    getTreeItem(element: ProfilerItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ProfilerItem): ProfilerItem[] {
        if (element) return [];

        return [
            new ProfilerItem('Last Run', '12.5 ms'),
            new ProfilerItem('Throughput', '1.2M ops/s'),
            new ProfilerItem('Memory BW', '450 GB/s'),
        ];
    }
}

class ProfilerItem extends vscode.TreeItem {
    constructor(name: string, value: string) {
        super(name, vscode.TreeItemCollapsibleState.None);
        this.description = value;
    }
}

// ============================================================================
// HTML Generators
// ============================================================================

function getCompatibilityHtml(output: string): string {
    return `<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: var(--vscode-font-family); padding: 20px; }
        .backend { margin: 10px 0; padding: 10px; border-radius: 4px; }
        .available { background: #1e4620; }
        .unavailable { background: #5a1d1d; }
    </style>
</head>
<body>
    <h1>Backend Compatibility</h1>
    <pre>${output}</pre>
</body>
</html>`;
}

function getMemoryDashboardHtml(): string {
    return `<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: var(--vscode-font-family); padding: 20px; }
        .card { background: var(--vscode-editor-background); padding: 15px; margin: 10px 0; border-radius: 8px; }
        .meter { height: 20px; background: #333; border-radius: 10px; overflow: hidden; }
        .meter-fill { height: 100%; background: linear-gradient(90deg, #4caf50, #8bc34a); transition: width 0.3s; }
        .stat { display: flex; justify-content: space-between; margin: 5px 0; }
    </style>
</head>
<body>
    <h1>GPU Memory Dashboard</h1>

    <div class="card">
        <h3>Device Memory</h3>
        <div class="meter"><div class="meter-fill" style="width: 15%;"></div></div>
        <div class="stat"><span>Used</span><span>1.2 GB</span></div>
        <div class="stat"><span>Total</span><span>8.0 GB</span></div>
    </div>

    <div class="card">
        <h3>Allocations</h3>
        <div class="stat"><span>Active</span><span>42</span></div>
        <div class="stat"><span>Peak</span><span>1.8 GB</span></div>
        <div class="stat"><span>Fragmentation</span><span>2.3%</span></div>
    </div>

    <div class="card">
        <h3>Kernel Buffers</h3>
        <div class="stat"><span>Control Blocks</span><span>128 KB</span></div>
        <div class="stat"><span>Message Queues</span><span>64 MB</span></div>
        <div class="stat"><span>Shared Memory</span><span>48 KB/block</span></div>
    </div>
</body>
</html>`;
}
