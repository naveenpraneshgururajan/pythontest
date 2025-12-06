import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Paper,
  TextField,
  Button,
  Grid,
  Alert,
  CircularProgress,
  Box,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Card,
  CardContent,
  Divider,
  Stack,
  ThemeProvider,
  createTheme,
} from '@mui/material';
import {
  BugReport,
  Search,
  Group,
  Psychology,
  Timeline,
  CheckCircle,
} from '@mui/icons-material';

// Lloyds Banking Group Theme
const lloydsTheme = createTheme({
  palette: {
    primary: {
      main: '#006A4D', // Lloyds dark green
      light: '#00845F',
      dark: '#00543E',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#019B8F', // Lloyds teal
      light: '#33AFA4',
      dark: '#017A71',
      contrastText: '#FFFFFF',
    },
    success: {
      main: '#4CAF50',
      light: '#81C784',
      dark: '#388E3C',
    },
    error: {
      main: '#D32F2F',
      light: '#EF5350',
      dark: '#C62828',
    },
    background: {
      default: '#F5F7F6',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#1A1A1A',
      secondary: '#5F6368',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h5: {
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    h6: {
      fontWeight: 600,
      letterSpacing: '-0.005em',
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '10px 24px',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 4px 8px rgba(0, 106, 77, 0.2)',
          },
        },
        contained: {
          '&:hover': {
            boxShadow: '0 4px 12px rgba(0, 106, 77, 0.3)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
        },
        elevation3: {
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 500,
        },
      },
    },
  },
});

function App() {
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [recentPredictions, setRecentPredictions] = useState([]);

  // Fetch model stats on component mount
  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/stats');
      const data = await response.json();
      setStats(data.modelInfo);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!description.trim() || description.length < 10) {
      setError('Please enter a bug description (at least 10 characters)');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ description }),
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
        // Add to recent predictions
        setRecentPredictions(prev => [{
          description: description,
          rootCause: data.prediction.rootCause.primary,
          fixTeam: data.prediction.fixTeam.primary,
          timestamp: new Date().toLocaleTimeString()
        }, ...prev.slice(0, 4)]);
      } else {
        setError(data.error || 'Failed to get prediction');
      }
    } catch (err) {
      setError('Failed to connect to server. Make sure the backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setDescription('');
    setResult(null);
    setError(null);
  };

  // Sample bug descriptions for testing
  const sampleBugs = [
    "Database connection timeout during peak hours",
    "Login button not responding on mobile app",
    "API returns 500 error when processing payment",
    "Customer data not syncing between CRM and billing system",
    "Search functionality showing duplicate results",
    "Password reset email not being delivered to users"
  ];

  const loadSampleBug = (sampleDescription) => {
    setDescription(sampleDescription);
    setError(null);
    setResult(null);
  };

  return (
    <ThemeProvider theme={lloydsTheme}>
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', bgcolor: 'background.default' }}>
        {/* Header */}
        <AppBar position="static" elevation={0} sx={{ bgcolor: 'primary.main' }}>
          <Container maxWidth="xl">
            <Toolbar sx={{ py: 1.5 }}>
              <BugReport sx={{ mr: 2, fontSize: 36 }} />
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="h5" component="h1" sx={{ fontWeight: 700, letterSpacing: '-0.02em' }}>
                  Bug Classification System
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9, mt: 0.5 }}>
                  AI-Powered Bug Triage | Lloyds Banking Group
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CheckCircle sx={{ fontSize: 20 }} />
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  Enterprise Ready
                </Typography>
              </Box>
            </Toolbar>
          </Container>
        </AppBar>

        {/* Main Content */}
        <Container maxWidth="xl" sx={{ flexGrow: 1, py: 5 }}>
          <Grid container spacing={4}>
            {/* Left Column - Input Form */}
            <Grid item xs={12} lg={5}>
              <Paper elevation={3} sx={{ p: 4, height: '100%', borderTop: '4px solid', borderColor: 'primary.main' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Psychology sx={{ mr: 1.5, color: 'primary.main', fontSize: 28 }} />
                  <Typography variant="h6" fontWeight={700} color="text.primary">
                    Bug Analysis
                  </Typography>
                </Box>

                <form onSubmit={handleSubmit}>
                  <TextField
                    fullWidth
                    multiline
                    rows={7}
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Describe the bug in detail... (minimum 10 characters)"
                    disabled={loading}
                    variant="outlined"
                    sx={{
                      mb: 1,
                      '& .MuiOutlinedInput-root': {
                        '&:hover fieldset': {
                          borderColor: 'primary.main',
                        },
                      }
                    }}
                  />

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                    <Typography variant="caption" color="text.secondary">
                      {description.length} characters
                    </Typography>
                    {description.length >= 10 && (
                      <Chip
                        label="Ready"
                        size="small"
                        color="success"
                        icon={<CheckCircle />}
                        sx={{ fontSize: '0.7rem' }}
                      />
                    )}
                  </Box>

                  <Stack direction="row" spacing={2} sx={{ mb: 4 }}>
                    <Button
                      type="submit"
                      variant="contained"
                      size="large"
                      fullWidth
                      disabled={loading || !description.trim()}
                      startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <Search />}
                      sx={{
                        py: 1.5,
                        fontSize: '1rem',
                      }}
                    >
                      {loading ? 'Analyzing...' : 'Analyze Bug'}
                    </Button>

                    <Button
                      type="button"
                      variant="outlined"
                      size="large"
                      onClick={handleClear}
                      disabled={loading}
                      sx={{ minWidth: '120px' }}
                    >
                      Clear
                    </Button>
                  </Stack>
                </form>

                {/* Sample Bugs */}
                <Divider sx={{ mb: 3 }} />
                <Typography variant="subtitle2" gutterBottom fontWeight={600} color="text.primary" sx={{ mb: 2 }}>
                  Sample Bug Reports
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                  {sampleBugs.map((bug, index) => (
                    <Button
                      key={index}
                      variant="outlined"
                      onClick={() => loadSampleBug(bug)}
                      disabled={loading}
                      sx={{
                        justifyContent: 'flex-start',
                        textAlign: 'left',
                        py: 1.5,
                        px: 2,
                        borderColor: 'divider',
                        color: 'text.primary',
                        textTransform: 'none',
                        fontWeight: 400,
                        '&:hover': {
                          borderColor: 'primary.main',
                          bgcolor: 'primary.50',
                        }
                      }}
                    >
                      {bug}
                    </Button>
                  ))}
                </Box>
              </Paper>
            </Grid>

            {/* Right Column - Results */}
            <Grid item xs={12} lg={7}>
              {/* Error Message */}
              {error && (
                <Alert severity="error" sx={{ mb: 3 }} icon={<BugReport />}>
                  <Typography variant="body2" fontWeight={500}>{error}</Typography>
                </Alert>
              )}

              {/* Loading State */}
              {loading && (
                <Paper elevation={3} sx={{ p: 6, textAlign: 'center' }}>
                  <CircularProgress size={70} thickness={4} />
                  <Typography variant="h6" sx={{ mt: 3, fontWeight: 600 }} color="primary">
                    Analyzing Bug Description
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Our AI model is processing your request...
                  </Typography>
                </Paper>
              )}

              {/* Prediction Results */}
              {result && result.success && (
                <Paper elevation={3} sx={{ p: 4, borderTop: '4px solid', borderColor: 'success.main' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                    <CheckCircle sx={{ mr: 1.5, color: 'success.main', fontSize: 28 }} />
                    <Typography variant="h6" fontWeight={700}>
                      Analysis Complete
                    </Typography>
                  </Box>

                  {/* Root Cause */}
                  <Box sx={{ mb: 4 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Search sx={{ mr: 1, color: 'primary.main', fontSize: 24 }} />
                      <Typography variant="subtitle1" fontWeight={600}>
                        Root Cause Classification
                      </Typography>
                    </Box>

                    <Card variant="outlined" sx={{ mb: 2, bgcolor: '#E8F5E9', borderColor: 'success.light', borderWidth: 2 }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                          <Chip
                            label={result.prediction.rootCause.primary}
                            color="primary"
                            size="medium"
                            sx={{ fontWeight: 600, fontSize: '0.95rem', py: 2.5 }}
                          />
                          <Box sx={{ textAlign: 'right' }}>
                            <Typography variant="h6" color="primary" fontWeight={700}>
                              {(result.prediction.rootCause.confidence * 100).toFixed(1)}%
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              Confidence
                            </Typography>
                          </Box>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={result.prediction.rootCause.confidence * 100}
                          sx={{ height: 10, borderRadius: 5 }}
                        />
                      </CardContent>
                    </Card>

                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5, fontWeight: 500 }}>
                      Alternative Classifications:
                    </Typography>
                    <Stack spacing={1}>
                      {result.prediction.rootCause.alternatives.slice(1).map((alt, index) => (
                        <Box
                          key={index}
                          sx={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            p: 1.5,
                            bgcolor: 'background.paper',
                            borderRadius: 1,
                            border: '1px solid',
                            borderColor: 'divider'
                          }}
                        >
                          <Typography variant="body2" fontWeight={500}>{alt.cause}</Typography>
                          <Typography variant="body2" color="text.secondary" fontWeight={600}>
                            {(alt.confidence * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                      ))}
                    </Stack>
                  </Box>

                  {/* Fix Team */}
                  <Box sx={{ mb: 4 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Group sx={{ mr: 1, color: 'secondary.main', fontSize: 24 }} />
                      <Typography variant="subtitle1" fontWeight={600}>
                        Recommended Fix Team
                      </Typography>
                    </Box>

                    <Card variant="outlined" sx={{ mb: 2, bgcolor: '#E0F7F4', borderColor: 'secondary.light', borderWidth: 2 }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                          <Chip
                            label={result.prediction.fixTeam.primary}
                            color="secondary"
                            size="medium"
                            sx={{ fontWeight: 600, fontSize: '0.95rem', py: 2.5 }}
                          />
                          <Box sx={{ textAlign: 'right' }}>
                            <Typography variant="h6" color="secondary" fontWeight={700}>
                              {(result.prediction.fixTeam.confidence * 100).toFixed(1)}%
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              Confidence
                            </Typography>
                          </Box>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={result.prediction.fixTeam.confidence * 100}
                          color="secondary"
                          sx={{ height: 10, borderRadius: 5 }}
                        />
                      </CardContent>
                    </Card>

                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5, fontWeight: 500 }}>
                      Alternative Teams:
                    </Typography>
                    <Stack spacing={1}>
                      {result.prediction.fixTeam.alternatives.slice(1).map((alt, index) => (
                        <Box
                          key={index}
                          sx={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            p: 1.5,
                            bgcolor: 'background.paper',
                            borderRadius: 1,
                            border: '1px solid',
                            borderColor: 'divider'
                          }}
                        >
                          <Typography variant="body2" fontWeight={500}>{alt.team}</Typography>
                          <Typography variant="body2" color="text.secondary" fontWeight={600}>
                            {(alt.confidence * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                      ))}
                    </Stack>
                  </Box>

                  {/* Model Performance */}
                  <Divider sx={{ mb: 3 }} />
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Timeline sx={{ mr: 1, color: 'success.main', fontSize: 24 }} />
                    <Typography variant="subtitle1" fontWeight={600}>
                      Model Performance Metrics
                    </Typography>
                  </Box>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Card sx={{ bgcolor: 'primary.50', border: '2px solid', borderColor: 'primary.light' }}>
                        <CardContent sx={{ textAlign: 'center', py: 3 }}>
                          <Typography variant="body2" color="text.secondary" fontWeight={500} gutterBottom>
                            Root Cause Accuracy
                          </Typography>
                          <Typography variant="h4" color="primary" fontWeight={700}>
                            {(result.modelAccuracy.rootCause * 100).toFixed(1)}%
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card sx={{ bgcolor: 'secondary.50', border: '2px solid', borderColor: 'secondary.light' }}>
                        <CardContent sx={{ textAlign: 'center', py: 3 }}>
                          <Typography variant="body2" color="text.secondary" fontWeight={500} gutterBottom>
                            Fix Team Accuracy
                          </Typography>
                          <Typography variant="h4" color="secondary" fontWeight={700}>
                            {(result.modelAccuracy.fixTeam * 100).toFixed(1)}%
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                </Paper>
              )}

              {/* Model Stats */}
              {stats && !result && !loading && (
                <Paper elevation={3} sx={{ p: 4, borderTop: '4px solid', borderColor: 'secondary.main' }}>
                  <Typography variant="h6" gutterBottom fontWeight={700} sx={{ mb: 3 }}>
                    Model Capabilities
                  </Typography>
                  <Grid container spacing={3}>
                    <Grid item xs={6}>
                      <Card sx={{ bgcolor: 'primary.50', textAlign: 'center', py: 3, border: '2px solid', borderColor: 'primary.light' }}>
                        <Typography variant="h3" color="primary.main" fontWeight={700}>
                          {stats.rootCauseCategories}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" fontWeight={500} sx={{ mt: 1 }}>
                          Root Cause Categories
                        </Typography>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card sx={{ bgcolor: 'secondary.50', textAlign: 'center', py: 3, border: '2px solid', borderColor: 'secondary.light' }}>
                        <Typography variant="h3" color="secondary.main" fontWeight={700}>
                          {stats.fixTeams}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" fontWeight={500} sx={{ mt: 1 }}>
                          Specialized Fix Teams
                        </Typography>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card sx={{ bgcolor: 'success.50', textAlign: 'center', py: 3, border: '2px solid', borderColor: 'success.light' }}>
                        <Typography variant="h3" color="success.main" fontWeight={700}>
                          {stats.modelAccuracy.rootCause}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" fontWeight={500} sx={{ mt: 1 }}>
                          Root Cause Accuracy
                        </Typography>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card sx={{ bgcolor: 'success.50', textAlign: 'center', py: 3, border: '2px solid', borderColor: 'success.light' }}>
                        <Typography variant="h3" color="success.main" fontWeight={700}>
                          {stats.modelAccuracy.fixTeam}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" fontWeight={500} sx={{ mt: 1 }}>
                          Fix Team Accuracy
                        </Typography>
                      </Card>
                    </Grid>
                  </Grid>
                </Paper>
              )}
            </Grid>
          </Grid>

          {/* Recent Predictions History */}
          {recentPredictions.length > 0 && (
            <Paper elevation={3} sx={{ mt: 4, p: 4, borderTop: '4px solid', borderColor: 'primary.main' }}>
              <Typography variant="h6" gutterBottom fontWeight={700} sx={{ mb: 3 }}>
                Recent Analysis History
              </Typography>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow sx={{ bgcolor: 'primary.50' }}>
                      <TableCell sx={{ fontWeight: 700 }}>Time</TableCell>
                      <TableCell sx={{ fontWeight: 700 }}>Bug Description</TableCell>
                      <TableCell sx={{ fontWeight: 700 }}>Root Cause</TableCell>
                      <TableCell sx={{ fontWeight: 700 }}>Fix Team</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {recentPredictions.map((pred, index) => (
                      <TableRow key={index} hover sx={{ '&:hover': { bgcolor: 'action.hover' } }}>
                        <TableCell sx={{ fontWeight: 500 }}>{pred.timestamp}</TableCell>
                        <TableCell sx={{ maxWidth: '300px' }}>
                          {pred.description.substring(0, 50)}...
                        </TableCell>
                        <TableCell>
                          <Chip label={pred.rootCause} size="small" color="primary" />
                        </TableCell>
                        <TableCell>
                          <Chip label={pred.fixTeam} size="small" color="secondary" />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          )}
        </Container>

        {/* Footer */}
        <Box
          component="footer"
          sx={{
            bgcolor: 'primary.dark',
            color: 'white',
            py: 3,
            mt: 'auto'
          }}
        >
          <Container maxWidth="xl">
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2" fontWeight={500}>
                Bug Classification System v1.0
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Lloyds Banking Group | Enterprise AI Solutions
              </Typography>
            </Box>
          </Container>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
