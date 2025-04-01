import './App.css';
import ImageUpload from './components/ImageUpload';
import Search from './components/Search';
import ClusterManagement from './components/ClusterManagement';
import BatchProcess from './components/BatchProcess';
import PhotoGallery from './components/PhotoGallery';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>FaceLens</h1>
        <BatchProcess />
        <ImageUpload />
        <ClusterManagement />
        <PhotoGallery />
        <Search />
      </header>
    </div>
  );
}

export default App;
