'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import LoadingSpinner from '@/components/LoadingSpinner';
import AuthService from '@/lib/auth';
import { apiService } from '@/lib/api';
import { User } from '@/types';

export default function SettingsPage() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    currentPassword: '',
    newPassword: '',
    confirmNewPassword: '',
  });

  useEffect(() => {
    // Check if user is authenticated
    if (!AuthService.isAuthenticated()) {
      router.push('/login');
      return;
    }

    // Load user data
    const user = AuthService.getUser();
    if (user) {
      setFormData(prevData => ({
        ...prevData,
        name: user.name || '',
        email: user.email || '',
      }));
    }
    
    setIsLoading(false);
  }, [router]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleProfileUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage('');
    setSuccessMessage('');
    
    try {
      setIsSaving(true);
      
      // In a real app, call the API to update profile
      // await apiService.user.updateProfile({
      //   name: formData.name,
      //   email: formData.email,
      // });
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Update local user data
      const currentUser = AuthService.getUser();
      if (currentUser) {
        const updatedUser: User = {
          ...currentUser,
          name: formData.name,
          email: formData.email,
        };
        AuthService.setUser(updatedUser);
      }
      
      setSuccessMessage('Profile updated successfully');
    } catch (error: any) {
      setErrorMessage(error.message || 'Failed to update profile');
    } finally {
      setIsSaving(false);
    }
  };

  const handlePasswordUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage('');
    setSuccessMessage('');
    
    // Validate password
    if (formData.newPassword !== formData.confirmNewPassword) {
      setErrorMessage('New passwords do not match');
      return;
    }
    
    if (formData.newPassword.length < 8) {
      setErrorMessage('Password must be at least 8 characters long');
      return;
    }
    
    try {
      setIsSaving(true);
      
      // In a real app, call the API to change password
      // await apiService.user.changePassword({
      //   currentPassword: formData.currentPassword,
      //   newPassword: formData.newPassword,
      // });
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 800));
      
      setSuccessMessage('Password changed successfully');
      
      // Clear password fields
      setFormData(prevData => ({
        ...prevData,
        currentPassword: '',
        newPassword: '',
        confirmNewPassword: '',
      }));
    } catch (error: any) {
      setErrorMessage(error.message || 'Failed to change password');
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteAccount = async () => {
    if (window.confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
      try {
        setIsLoading(true);
        
        // In a real app, call the API to delete account
        // await apiService.user.deleteAccount();
        
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 800));
        
        // Log out the user
        AuthService.clearAuth();
        router.push('/');
      } catch (error: any) {
        setErrorMessage(error.message || 'Failed to delete account');
        setIsLoading(false);
      }
    }
  };

  if (isLoading) {
    return (
      <main className="flex min-h-screen flex-col">
        <Header />
        <div className="flex-grow flex items-center justify-center">
          <LoadingSpinner size="large" text="Loading your settings..." />
        </div>
        <Footer />
      </main>
    );
  }

  return (
    <main className="flex min-h-screen flex-col">
      <Header />
      
      <div className="container mx-auto px-4 py-8 flex-grow">
        <h1 className="text-3xl font-bold mb-6">Account Settings</h1>
        
        {successMessage && (
          <div className="bg-green-100 border border-green-500 text-green-700 px-4 py-3 rounded mb-6">
            {successMessage}
          </div>
        )}
        
        {errorMessage && (
          <div className="bg-red-100 border border-red-500 text-red-700 px-4 py-3 rounded mb-6">
            {errorMessage}
          </div>
        )}
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-8">
            {/* Profile Settings Form */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Profile Information</h2>
              <form onSubmit={handleProfileUpdate} className="space-y-4">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                    Full Name
                  </label>
                  <input
                    id="name"
                    name="name"
                    type="text"
                    value={formData.name}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                    Email Address
                  </label>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    value={formData.email}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                
                <div className="pt-4">
                  <button 
                    type="submit" 
                    className="btn-primary px-4 py-2" 
                    disabled={isSaving}
                  >
                    {isSaving ? 'Saving...' : 'Save Changes'}
                  </button>
                </div>
              </form>
            </div>
            
            {/* Password Change Form */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Change Password</h2>
              <form onSubmit={handlePasswordUpdate} className="space-y-4">
                <div>
                  <label htmlFor="currentPassword" className="block text-sm font-medium text-gray-700 mb-1">
                    Current Password
                  </label>
                  <input
                    id="currentPassword"
                    name="currentPassword"
                    type="password"
                    value={formData.currentPassword}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                
                <div>
                  <label htmlFor="newPassword" className="block text-sm font-medium text-gray-700 mb-1">
                    New Password
                  </label>
                  <input
                    id="newPassword"
                    name="newPassword"
                    type="password"
                    value={formData.newPassword}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    minLength={8}
                    required
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    Password must be at least 8 characters long
                  </p>
                </div>
                
                <div>
                  <label htmlFor="confirmNewPassword" className="block text-sm font-medium text-gray-700 mb-1">
                    Confirm New Password
                  </label>
                  <input
                    id="confirmNewPassword"
                    name="confirmNewPassword"
                    type="password"
                    value={formData.confirmNewPassword}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    minLength={8}
                    required
                  />
                </div>
                
                <div className="pt-4">
                  <button 
                    type="submit" 
                    className="btn-primary px-4 py-2" 
                    disabled={isSaving}
                  >
                    {isSaving ? 'Changing Password...' : 'Change Password'}
                  </button>
                </div>
              </form>
            </div>
          </div>
          
          <div className="lg:col-span-1">
            {/* Danger Zone */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold text-red-600 mb-4">Danger Zone</h2>
              <p className="text-gray-600 mb-6">
                These actions cannot be undone. Please proceed with caution.
              </p>
              <button
                onClick={handleDeleteAccount}
                className="w-full bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded transition duration-200"
                disabled={isLoading}
              >
                Delete Account
              </button>
            </div>
            
            {/* Account Information */}
            <div className="bg-white rounded-lg shadow p-6 mt-6">
              <h2 className="text-xl font-semibold mb-4">Account Information</h2>
              <div className="text-gray-600 space-y-2">
                <p>
                  <span className="font-medium">Account Type:</span>
                  {' '}Free
                </p>
                <p>
                  <span className="font-medium">Member Since:</span>
                  {' '}June 15, 2025
                </p>
                <p>
                  <span className="font-medium">Matching Credits:</span>
                  {' '}Unlimited
                </p>
              </div>
              <div className="mt-6">
                <button
                  onClick={() => router.push('/upgrade')}
                  className="w-full bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-600 hover:to-indigo-700 text-white font-medium py-2 px-4 rounded transition duration-200"
                >
                  Upgrade Account
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <Footer />
    </main>
  );
}
