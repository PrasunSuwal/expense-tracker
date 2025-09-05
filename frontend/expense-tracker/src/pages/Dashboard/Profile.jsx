import React, { useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import { useUserAuth } from "../../hooks/useUserAuth";
import DashboardLayout from "../../components/layouts/DashboardLayout";
import { UserContext } from "../../context/UserContext";
import ProfilePhotoSelector from "../../components/Inputs/ProfilePhotoSelector";
import Input from "../../components/Inputs/Input";
import axiosInstance from "../../utils/axiosInstance";
import { API_PATHS } from "../../utils/apiPaths";
import uploadImage from "../../utils/uploadImage";
import toast from "react-hot-toast";

const Profile = () => {
  useUserAuth();
  const navigate = useNavigate();
  const { user, updateUser } = useContext(UserContext);

  const [fullName, setFullName] = useState(user?.fullName || "");
  const [profilePic, setProfilePic] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpdateProfile = async (e) => {
    e.preventDefault();

    if (!fullName.trim()) {
      toast.error("Full name is required");
      return;
    }

    setLoading(true);

    try {
      let profileImageUrl = user?.profileImageUrl || "";

      // Upload new image if selected
      if (profilePic) {
        const imgUploadRes = await uploadImage(profilePic);
        profileImageUrl = imgUploadRes.imageUrl || "";
      }

      const response = await axiosInstance.put(API_PATHS.AUTH.UPDATE_PROFILE, {
        fullName,
        profileImageUrl,
      });

      if (response.data) {
        updateUser(response.data);
        toast.success("Profile updated successfully!");
        setProfilePic(null); // Clear the selected image
      }
    } catch (error) {
      console.error("Error updating profile:", error);
      if (error.response && error.response.data.message) {
        toast.error(error.response.data.message);
      } else {
        toast.error("Something went wrong. Please try again");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-2xl mx-auto">
        <div className="card">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Profile Settings
            </h2>
            <p className="text-gray-600">Update your profile information</p>
          </div>

          <form onSubmit={handleUpdateProfile}>
            <div className="mb-8">
              <label className="block text-sm font-medium text-gray-700 mb-4">
                Profile Picture
              </label>
              <ProfilePhotoSelector
                image={profilePic}
                setImage={setProfilePic}
                currentImageUrl={user?.profileImageUrl}
              />
            </div>

            <div className="mb-6">
              <Input
                value={fullName}
                onChange={({ target }) => setFullName(target.value)}
                label="Full Name"
                placeholder="Enter your full name"
                type="text"
              />
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email Address <span className="text-red-500">*</span>
              </label>
              <input
                value={user?.email || ""}
                placeholder="Email address"
                type="email"
                disabled
                onClick={() => toast.error("Email cannot be changed")}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-purple-500 focus:border-purple-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
              />
              <p className="text-xs text-gray-500 mt-1">
                Email cannot be changed
              </p>
            </div>

            <div className="flex justify-end gap-4">
              <button
                type="button"
                className="px-6 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
                onClick={() => {
                  navigate("/dashboard");
                }}
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading}
                className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Updating..." : "Update Profile"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default Profile;
