import React, { useState } from 'react';
import { Container } from './components/layout/Container';
import { Button } from './components/ui/Button';
import { Github, Server } from 'lucide-react';
import { api, Item } from './lib/api';
import { useQuery, useMutation, QueryClient, QueryClientProvider } from 'react-query';

const queryClient = new QueryClient();

function AppContent() {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const { data: items = [] } = useQuery('items', api.items.list);

  const createItem = useMutation(api.items.create, {
    onSuccess: () => {
      queryClient.invalidateQueries('items');
      setName('');
      setDescription('');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    createItem.mutate({ name, description });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Container className="py-12">
        <div className="text-center">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
            React + Python Base Directory
          </h1>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            A clean starting point for your web applications with React, TypeScript, and Python FastAPI backend.
          </p>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            <Button size="lg">
              Get Started
            </Button>
            <Button variant="outline" size="lg">
              <Github className="mr-2 h-5 w-5" />
              View on GitHub
            </Button>
          </div>

          <div className="mt-16">
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="flex items-center justify-center gap-2 text-gray-600 mb-6">
                <Server className="h-5 w-5" />
                <span>API Test Interface</span>
              </div>

              <form onSubmit={handleSubmit} className="max-w-md mx-auto">
                <div className="space-y-4">
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                      Name
                    </label>
                    <input
                      type="text"
                      id="name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                      required
                    />
                  </div>
                  <div>
                    <label htmlFor="description" className="block text-sm font-medium text-gray-700">
                      Description
                    </label>
                    <textarea
                      id="description"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                      rows={3}
                    />
                  </div>
                  <Button type="submit" className="w-full">
                    Add Item
                  </Button>
                </div>
              </form>

              <div className="mt-8">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Items List</h3>
                <div className="space-y-4">
                  {items.map((item: Item) => (
                    <div key={item.id} className="bg-gray-50 p-4 rounded-md">
                      <h4 className="font-medium text-gray-900">{item.name}</h4>
                      {item.description && (
                        <p className="mt-1 text-sm text-gray-500">{item.description}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </Container>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}

export default App;