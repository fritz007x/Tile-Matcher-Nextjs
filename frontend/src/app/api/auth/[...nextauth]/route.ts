import NextAuth from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import CredentialsProvider from "next-auth/providers/credentials";
import axios from "axios";

// Environment variable validation
const NEXT_PUBLIC_API_URL = process.env.NEXT_PUBLIC_API_URL;
const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;

if (!NEXT_PUBLIC_API_URL) {
  console.error('Missing NEXT_PUBLIC_API_URL environment variable');
}

// Optional Google provider - only enabled if credentials are provided
const useGoogleProvider = GOOGLE_CLIENT_ID && GOOGLE_CLIENT_SECRET;

const handler = NextAuth({
  providers: [
    // Only include Google provider if credentials are available
    ...(useGoogleProvider ? [
      GoogleProvider({
        clientId: GOOGLE_CLIENT_ID as string,
        clientSecret: GOOGLE_CLIENT_SECRET as string,
      })
    ] : []),
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) return null;
        
        try {
          if (!NEXT_PUBLIC_API_URL) {
            throw new Error('API URL not configured');
          }
          
          const loginUrl = `${NEXT_PUBLIC_API_URL}/api/token`;
          console.log('Attempting login to:', loginUrl);
          
          // Call the backend API for authentication
          const response = await axios.post(loginUrl, {
            username: credentials.email,
            password: credentials.password,
          });
          
          if (response.data) {
            // Return user object which will be stored in the JWT
            return {
              id: response.data.user_id || response.data.id,
              name: response.data.name || response.data.username,
              email: response.data.email,
              accessToken: response.data.access_token,
            };
          }
          return null;
        } catch (error) {
          console.error("Authentication error:", error);
          // Provide more specific error information for debugging
          if (axios.isAxiosError(error)) {
            console.error('API response:', error.response?.data);
          }
          return null;
        }
      }
    })
  ],
  callbacks: {
    async jwt({ token, user, account }) {
      // Initial sign in
      if (account && user) {
        return {
          ...token,
          accessToken: user.accessToken,
          id: user.id,
        };
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = token.id as string;
        session.user.accessToken = token.accessToken as string;
      }
      return session;
    }
  },
  pages: {
    signIn: "/login",
    error: "/login",
  },
  session: {
    strategy: "jwt",
    maxAge: 30 * 24 * 60 * 60, // 30 days
  },
});

export { handler as GET, handler as POST };
