document.addEventListener('DOMContentLoaded', () => {
    // Navigation Logic
    const navButtons = document.querySelectorAll('.nav-btn');
    const views = {
        'COCKPIT': document.getElementById('view-cockpit'),
        'EVALUATION': document.getElementById('view-evaluation'),
        'SYSTEM': document.getElementById('view-system')
    };
    const pageTitle = document.getElementById('page-title');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.getAttribute('data-tab');

            // Update Active State
            navButtons.forEach(b => {
                const isActive = b === btn;
                const indicator = b.querySelector('.indicator');

                if (isActive) {
                    // Active Styles
                    b.classList.remove('bg-neutral-800', 'text-green-600', 'border-neutral-700', 'hover:bg-neutral-700', 'hover:translate-x-1');
                    b.classList.add('bg-green-600', 'text-black', 'border-green-400', 'shadow-[4px_4px_0px_0px_#000]');
                    if (indicator) indicator.classList.remove('hidden');
                } else {
                    // Inactive Styles
                    b.classList.add('bg-neutral-800', 'text-green-600', 'border-neutral-700', 'hover:bg-neutral-700', 'hover:translate-x-1');
                    b.classList.remove('bg-green-600', 'text-black', 'border-green-400', 'shadow-[4px_4px_0px_0px_#000]');
                    if (indicator) indicator.classList.add('hidden');
                }
            });

            // Switch View
            Object.values(views).forEach(view => {
                if (view) view.classList.add('hidden');
            });
            if (views[tabName]) {
                views[tabName].classList.remove('hidden');
            }
        });
    });

    // Run Evaluation Logic
    const runForm = document.getElementById('run-form');
    const terminalOutput = document.getElementById('terminal-output');
    const btnStart = document.getElementById('btn-start');

    if (runForm) {
        runForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData(runForm);
            const data = Object.fromEntries(formData.entries());

            // Convert types
            if (data.limit) data.limit = parseInt(data.limit);
            if (data.max_connections) data.max_connections = parseInt(data.max_connections);
            if (data.max_tokens) data.max_tokens = parseInt(data.max_tokens);
            if (data.temperature) data.temperature = parseFloat(data.temperature);

            // Update Button State
            const originalBtnText = btnStart.textContent;
            btnStart.textContent = 'PROCESSING...';
            btnStart.classList.remove('bg-blue-600', 'border-blue-800', 'hover:bg-blue-500');
            btnStart.classList.add('bg-yellow-500', 'border-yellow-700', 'text-black');

            // Clear Terminal or Add Separator
            const lineCount = terminalOutput.children.length + 1;
            const separator = document.createElement('div');
            separator.className = 'break-words mt-4 mb-2 text-yellow-500';
            separator.innerHTML = `<span class="text-green-700 mr-2">${lineCount.toString().padStart(2, '0')}</span>> INITIATING SEQUENCE...`;
            terminalOutput.appendChild(separator);

            try {
                // Start Stream
                const queryParams = new URLSearchParams(data).toString();
                const eventSource = new EventSource(`/stream?${queryParams}`);

                eventSource.onmessage = (event) => {
                    const currentCount = terminalOutput.children.length + 1;
                    const line = document.createElement('div');
                    line.className = 'break-words';

                    // Style based on content
                    let contentClass = 'text-green-500';
                    if (event.data.includes('[ERROR]')) contentClass = 'text-red-500';
                    if (event.data.includes('[DONE]')) contentClass = 'text-blue-400';

                    line.innerHTML = `<span class="text-green-700 mr-2">${currentCount.toString().padStart(2, '0')}</span><span class="${contentClass}">${event.data}</span>`;

                    terminalOutput.appendChild(line);
                    terminalOutput.scrollTop = terminalOutput.scrollHeight;

                    if (event.data.includes('[DONE]')) {
                        eventSource.close();
                        resetButton();
                    }
                };

                eventSource.onerror = (err) => {
                    console.error('EventSource failed:', err);
                    eventSource.close();
                    const currentCount = terminalOutput.children.length + 1;
                    const line = document.createElement('div');
                    line.className = 'break-words text-red-500';
                    line.innerHTML = `<span class="text-green-700 mr-2">${currentCount.toString().padStart(2, '0')}</span>CONNECTION LOST.`;
                    terminalOutput.appendChild(line);
                    resetButton();
                };

            } catch (error) {
                console.error('Error starting run:', error);
                const currentCount = terminalOutput.children.length + 1;
                const line = document.createElement('div');
                line.className = 'break-words text-red-500';
                line.innerHTML = `<span class="text-green-700 mr-2">${currentCount.toString().padStart(2, '0')}</span>ERROR: ${error.message}`;
                terminalOutput.appendChild(line);
                resetButton();
            }

            function resetButton() {
                btnStart.textContent = originalBtnText;
                btnStart.classList.add('bg-blue-600', 'border-blue-800', 'hover:bg-blue-500');
                btnStart.classList.remove('bg-yellow-500', 'border-yellow-700', 'text-black');
            }
        });
    }

    // Load Logs Logic
    const loadLogsBtn = document.getElementById('btn-load-logs');
    const logsList = document.getElementById('logs-list');

    if (loadLogsBtn) {
        loadLogsBtn.addEventListener('click', async () => {
            try {
                logsList.innerHTML = '<div class="text-green-500 animate-pulse">ACCESSING ARCHIVES...</div>';
                const response = await fetch('/api/logs');
                const logs = await response.json();

                logsList.innerHTML = '';
                if (logs.length === 0) {
                    logsList.innerHTML = '<div class="text-neutral-500">NO LOGS FOUND.</div>';
                    return;
                }

                logs.forEach(log => {
                    const item = document.createElement('div');
                    item.className = 'border-2 border-neutral-800 bg-black p-4 hover:border-green-600 cursor-pointer transition-colors group';
                    item.innerHTML = `
                        <div class="flex justify-between items-center">
                            <span class="text-green-400 font-terminal text-lg group-hover:text-green-300">${log.name}</span>
                            <span class="text-neutral-500 text-xs font-pixel">${new Date(log.modified * 1000).toLocaleString()}</span>
                        </div>
                    `;
                    item.onclick = () => viewLogDetails(log.name);
                    logsList.appendChild(item);
                });
            } catch (error) {
                console.error('Error loading logs:', error);
                logsList.innerHTML = '<div class="text-red-500">ERROR ACCESSING ARCHIVES.</div>';
            }
        });
    }

    async function viewLogDetails(logName) {
        try {
            const response = await fetch(`/api/logs/${logName}`);
            const data = await response.json();
            // Simple alert for now, or could replace the list with details
            alert(`LOG DATA RETRIEVED: ${logName}\n(Check console for full JSON dump)`);
            console.log(data);
        } catch (error) {
            console.error('Error loading log details:', error);
        }
    }
});
